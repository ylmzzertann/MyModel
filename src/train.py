"""
Model eğitimi.

    python -m src.train                       # tam akış
    python -m src.train --test-from 2025-07-01
    python -m src.train --reuse-dataset       # özellik tablosunu önbellekten al (hızlı deneme)
    python -m src.train --refit-all           # testten sonra TÜM veriyle yeniden eğit

Akış
----
1. Veri + özellikler (FeatureStore). Elo / piyasa Elo parametreleri YALNIZCA ilk doğrulama sezonu
   öncesi veriyle ayarlanır. Sızıntı ve eğitim/tahmin tutarlılık kontrolleri.
2. Zaman bazlı ayrım: geliştirme (< test_from) ve test (>= test_from). ASLA rastgele split yok.
3. Kayan-başlangıçlı çapraz doğrulama (rolling-origin CV): geliştirme dönemindeki son CV_FOLDS sezonun her
   biri sırayla doğrulama sezonu olur; model yalnızca o sezondan ÖNCEKİ verilerle eğitilir (erken durdurma
   için eğitim döneminin son sezonu ayrılır). Tüm kararlar biriken fold-dışı (out-of-fold) tahminlerle verilir:
     a) özellik grubu ileri seçimi
     b) XGBoost sınıflandırıcı hiperparametreleri (+ zaman ağırlığı)
     c) Lojistik Regresyon (baseline)
     d) Poisson skor modelleri (ev / deplasman golü) + Dixon-Coles rho
     e) topluluk ağırlıkları ve sıcaklık kalibrasyonu
   Ölçüt: 0.5 * log-loss(tüm maçlar) + 0.5 * log-loss(UEFA maçları)
4. Son modeller tüm geliştirme verisiyle eğitilir; test setinde bir kez değerlendirilir.
5. models/mac_modeli.pkl (joblib) ve feature_schema.md yazılır.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
from datetime import date, datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import data_collection as dc
from . import features as ft
from .predict import MAX_GOALS, combine, component_outputs, dixon_coles_matrix, wdl_from_matrix

log = logging.getLogger(__name__)

MODELS_DIR = dc.PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "mac_modeli.pkl"
SCHEMA_PATH = dc.PROJECT_ROOT / "feature_schema.md"
METRICS_PATH = MODELS_DIR / "metrics.json"
CALIBRATION_PATH = MODELS_DIR / "calibration_report.csv"
ELO_TUNING_PATH = MODELS_DIR / "elo_tuning.csv"
FEATURE_SELECTION_PATH = MODELS_DIR / "feature_selection.csv"
HYPERPARAM_PATH = MODELS_DIR / "hyperparameter_search.csv"
DATASET_CACHE_PATH = dc.PROCESSED_DIR / "train_dataset.joblib"

SCHEMA_JSON_BEGIN = "<!-- SCHEMA_JSON_BEGIN -->"
SCHEMA_JSON_END = "<!-- SCHEMA_JSON_END -->"
RANDOM_STATE = 42
CV_FOLDS = 4
SELECTION_MIN_IMPROVEMENT = 0.0003
N_HYPERPARAM_CONFIGS = 10
UEFA_MAIN = {f"uefa:{c}" for c in dc.UEFA_MAIN_COMPETITIONS}


# =============================================================================
# Metrikler
# =============================================================================

def multiclass_brier(y: np.ndarray, proba: np.ndarray) -> float:
    """Çok sınıflı Brier: satır başına sum_k (p_k - 1[y=k])^2 ortalaması (0 en iyi, 2 en kötü)."""
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(y: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, proba.argmax(axis=1))),
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier": multiclass_brier(y, proba),
    }


def objective(y: np.ndarray, proba: np.ndarray, uefa: np.ndarray) -> tuple[float, float, float]:
    """Karar ölçütü: 0.5 * log-loss(tüm) + 0.5 * log-loss(UEFA)."""
    ll_all = log_loss(y, proba, labels=[0, 1, 2])
    ll_uefa = log_loss(y[uefa], proba[uefa], labels=[0, 1, 2])
    return 0.5 * (ll_all + ll_uefa), ll_all, ll_uefa


def calibration_table(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Güvenilirlik (reliability) tablosu.
    - Her sınıf için: tahmin edilen olasılık kovası -> o sınıfın gerçekleşme oranı
    - 'top_label': modelin en olası dediği sonuç ve verdiği olasılık -> gerçekten tutma oranı
      (ör. "%70 dediği maçların gerçekten ~%70'i tutuyor mu?")
    """
    rows = []
    targets = {label: (proba[:, k], (y == k).astype(float)) for k, label in enumerate(ft.CLASS_LABELS)}
    targets["top_label"] = (proba.max(axis=1), (proba.argmax(axis=1) == y).astype(float))
    edges = np.linspace(0, 1, n_bins + 1)
    for target, (p, hit) in targets.items():
        bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            mask = bins == b
            if not mask.any():
                continue
            rows.append({"target": target, "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "count": int(mask.sum()),
                         "mean_predicted": float(p[mask].mean()), "observed_rate": float(hit[mask].mean())})
    df = pd.DataFrame(rows)
    df["gap"] = df["observed_rate"] - df["mean_predicted"]
    return df


def expected_calibration_error(table: pd.DataFrame) -> dict:
    return {target: float(np.sum(g["count"] / g["count"].sum() * g["gap"].abs()))
            for target, g in table.groupby("target")}


# =============================================================================
# Modeller
# =============================================================================

def time_weights(dates: pd.Series, ref: pd.Timestamp, half_life_years: float | None) -> np.ndarray | None:
    """Zaman ağırlığı: 0.5 ** (yaş_yıl / yarı_ömür); None ise ağırlıksız."""
    if half_life_years is None:
        return None
    age = (ref - dates).dt.days.to_numpy() / 365.25
    return np.power(0.5, age / half_life_years)


def make_xgb_classifier(p: dict, n_estimators: int, early_stopping: bool) -> xgb.XGBClassifier:
    # XGBoost NaN değerleri kendi içinde işler (her bölünmede öğrenilen varsayılan yön).
    return xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=n_estimators, learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], min_child_weight=p["min_child_weight"], subsample=0.8,
        colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], tree_method="hist",
        eval_metric="mlogloss", early_stopping_rounds=100 if early_stopping else None,
        random_state=RANDOM_STATE, n_jobs=-1)


def make_xgb_poisson(p: dict, n_estimators: int, early_stopping: bool) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="count:poisson", n_estimators=n_estimators, learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], min_child_weight=p["min_child_weight"], subsample=0.8,
        colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], tree_method="hist",
        eval_metric="poisson-nloglik", early_stopping_rounds=100 if early_stopping else None,
        random_state=RANDOM_STATE, n_jobs=-1)


def make_logreg(C: float) -> Pipeline:
    # Eksik değerler eğitim medyanıyla doldurulur; imputer pipeline'ın içinde olduğu için pkl ile taşınır.
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=5000))])


def fit_rho(lam_h: np.ndarray, lam_a: np.ndarray, goals_h: np.ndarray, goals_a: np.ndarray) -> float:
    """Dixon-Coles rho: gerçekleşen skorların log-olabilirliğini maksimize eden değer."""
    gh = np.minimum(goals_h, MAX_GOALS).astype(int)
    ga = np.minimum(goals_a, MAX_GOALS).astype(int)
    idx = np.arange(len(gh))

    def nll(rho):
        m = dixon_coles_matrix(lam_h, lam_a, rho)
        return -np.mean(np.log(np.clip(m[idx, gh, ga], 1e-12, None)))

    return float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x)


# =============================================================================
# Kayan-başlangıçlı çapraz doğrulama
# =============================================================================

def make_folds(test_from: pd.Timestamp, n: int = CV_FOLDS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """test_from öncesindeki son n sezon: [(doğrulama başı, doğrulama sonu), ...] (eskiden yeniye)."""
    return [(pd.Timestamp(test_from.year - k, 7, 1), pd.Timestamp(test_from.year - k + 1, 7, 1))
            for k in range(n, 0, -1)]


class RollingCV:
    def __init__(self, dev: pd.DataFrame, folds: list[tuple[pd.Timestamp, pd.Timestamp]]):
        self.splits = []
        for val_start, val_end in folds:
            stop_cut = pd.Timestamp(val_start.year - 1, 7, 1)
            train = dev[dev["date"] < val_start]
            self.splits.append({
                "val_start": val_start,
                "fit": train[train["date"] < stop_cut],
                "stop": train[train["date"] >= stop_cut],
                "val": dev[(dev["date"] >= val_start) & (dev["date"] < val_end)],
            })
        val = pd.concat([s["val"] for s in self.splits])
        self.y = val["target"].to_numpy()
        self.uefa = val["is_uefa"].to_numpy(float) == 1.0
        self.goals_h = val["home_goals"].to_numpy()
        self.goals_a = val["away_goals"].to_numpy()

    def score(self, proba: np.ndarray) -> tuple[float, float, float]:
        return objective(self.y, proba, self.uefa)

    def xgb_classifier(self, cols: list[str], p: dict) -> tuple[np.ndarray, list[int]]:
        oof, iters = [], []
        for s in self.splits:
            m = make_xgb_classifier(p, 3000, early_stopping=True)
            m.fit(s["fit"][cols], s["fit"]["target"], eval_set=[(s["stop"][cols], s["stop"]["target"])],
                  sample_weight=time_weights(s["fit"]["date"], s["val_start"], p["half_life"]), verbose=False)
            oof.append(m.predict_proba(s["val"][cols]))
            iters.append(int(m.best_iteration) + 1)
        return np.vstack(oof), iters

    def logreg(self, cols: list[str], C: float, half_life: float | None) -> np.ndarray:
        oof = []
        for s in self.splits:
            trainval = pd.concat([s["fit"], s["stop"]])
            m = make_logreg(C).fit(trainval[cols], trainval["target"],
                                   clf__sample_weight=time_weights(trainval["date"], s["val_start"], half_life))
            oof.append(m.predict_proba(s["val"][cols]))
        return np.vstack(oof)

    def poisson(self, cols: list[str], p: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
        lam = {"home_goals": [], "away_goals": []}
        iters = []
        for s in self.splits:
            w = time_weights(s["fit"]["date"], s["val_start"], p["half_life"])
            for target in lam:
                m = make_xgb_poisson(p, 3000, early_stopping=True)
                m.fit(s["fit"][cols], s["fit"][target], eval_set=[(s["stop"][cols], s["stop"][target])],
                      sample_weight=w, verbose=False)
                lam[target].append(m.predict(s["val"][cols]))
                iters.append(int(m.best_iteration) + 1)
        return np.concatenate(lam["home_goals"]), np.concatenate(lam["away_goals"]), iters


def select_feature_groups(cv: RollingCV, groups: dict[str, list[str]]) -> tuple[list[str], pd.DataFrame]:
    """
    İleri seçim (forward selection): 'base' grubu her zaman vardır; her turda CV ölçütünü en çok
    iyileştiren grup eklenir, iyileşme SELECTION_MIN_IMPROVEMENT'tan küçükse durulur.
    Model: XGBoost (derinlik 4, lr 0.05), zaman ağırlığı yok. Test seti KULLANILMAZ.
    """
    probe = {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 20, "colsample_bytree": 0.8,
             "reg_lambda": 2.0, "half_life": None}

    def score(cols):
        return cv.score(cv.xgb_classifier(cols, probe)[0])

    selected = list(groups["base"])
    best, ll_all, ll_uefa = score(selected)
    rows = [{"round": 0, "group": "base", "objective": best, "cv_log_loss": ll_all, "cv_uefa_log_loss": ll_uefa,
             "added": True}]
    log.info("  [tur 0] base: ölçüt=%.5f (tüm %.5f, UEFA %.5f)", best, ll_all, ll_uefa)
    remaining = [g for g in groups if g != "base"]
    rnd = 0
    while remaining:
        rnd += 1
        trials = {}
        for g in remaining:
            trials[g] = score(selected + groups[g])
            log.info("  [tur %d] +%-12s ölçüt=%.5f (tüm %.5f, UEFA %.5f)", rnd, g, *trials[g])
        g_best = min(trials, key=lambda g: trials[g][0])
        improved = trials[g_best][0] < best - SELECTION_MIN_IMPROVEMENT
        for g, (obj, a, u) in trials.items():
            rows.append({"round": rnd, "group": g, "objective": obj, "cv_log_loss": a, "cv_uefa_log_loss": u,
                         "added": improved and g == g_best})
        if not improved:
            log.info("  Durdu: en iyi aday (+%s) ölçütü %.5f'ten yeterince iyileştirmedi.", g_best, best)
            break
        selected += groups[g_best]
        best = trials[g_best][0]
        remaining.remove(g_best)
        log.info("  -> eklendi: %s (ölçüt %.5f)", g_best, best)
    return [c for c in ft.FEATURE_COLUMNS if c in selected], pd.DataFrame(rows)


def search_xgb_classifier(cv: RollingCV, cols: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Rastgele hiperparametre araması (sabit tohum); en iyi config, tüm denemeler ve en iyinin OOF tahmini."""
    grid = {"max_depth": [3, 4, 5], "min_child_weight": [20, 50, 100], "colsample_bytree": [0.6, 0.8],
            "reg_lambda": [2.0, 10.0], "half_life": [None, 4.0, 8.0]}
    all_configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    rng = np.random.default_rng(RANDOM_STATE)
    configs = [{"max_depth": 4, "min_child_weight": 20, "colsample_bytree": 0.8, "reg_lambda": 2.0, "half_life": None}]
    configs += [all_configs[i] for i in rng.choice(len(all_configs), N_HYPERPARAM_CONFIGS - 1, replace=False)]
    rows, best = [], None
    for c in configs:
        p = {"learning_rate": 0.03, **c}
        oof, iters = cv.xgb_classifier(cols, p)
        obj, a, u = cv.score(oof)
        rows.append({**p, "objective": obj, "cv_log_loss": a, "cv_uefa_log_loss": u, "iterations": iters})
        log.info("  xgb %s -> ölçüt=%.5f (tüm %.5f, UEFA %.5f)", c, obj, a, u)
        if best is None or obj < best[0]:
            best = (obj, {**p, "n_estimators": int(np.median(iters) * 1.1)}, oof)
    return best[1], pd.DataFrame(rows).sort_values("objective"), best[2]


# =============================================================================
# Veri hazırlama
# =============================================================================

def default_test_from(today: date) -> pd.Timestamp:
    """Test = son tamamlanmış sezonun başından bugüne (devam eden sezon dahil)."""
    return pd.Timestamp(dc.season_start_year(today) - 1, 7, 1)


def prepare_dataset(tune_until: pd.Timestamp, reuse: bool = False) -> tuple[pd.DataFrame, "ft.FeatureStore", dict]:
    signature = {"data": ft.data_signature(), "tune_until": str(tune_until.date()),
                 "params": ft.contract_hash({"p": ft.FEATURE_PARAMS, "f": ft.FEATURE_SPECS})}
    if reuse and DATASET_CACHE_PATH.exists():
        cached = joblib.load(DATASET_CACHE_PATH)
        if cached["signature"] == signature:
            log.info("Özellik tablosu önbellekten alındı: %s", DATASET_CACHE_PATH)
            return cached["data"], cached["store"], cached["ctx"]
        log.info("Önbellek güncel değil (veri ya da özellik tanımları değişti); yeniden hesaplanıyor.")

    store, ctx = ft.FeatureStore.build(elo_params=None, tune_until=tune_until)
    matches = ctx["matches"]
    log.info("Seçilen Elo parametreleri: %s", store.elo_params)

    dc.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = ctx["identity_report"]
    report.to_csv(dc.REPORTS_DIR / "team_identity.csv", index=False)
    log.info("Takım kimliği eşleştirmesi: %s", report["method"].value_counts().to_dict())

    feats = store.compute(matches[["date", "home_key", "away_key", "neutral", "is_knockout", "is_uefa"]])
    ft.assert_form_has_no_leakage(matches, feats)
    ft.assert_elo_consistency(matches, feats)
    ft.assert_player_state_no_leakage(matches, feats, ctx["appearances"], ctx["valuations"], ctx["value_index"])
    log.info("Sızıntı ve eğitim/tahmin tutarlılık kontrolleri geçti (form, gol formu, şut payı, Elo, "
             "piyasa Elo'su, oyuncu durumu).")

    # is_knockout / is_uefa hem maç tablosunda hem özelliklerde var; özellik tablosundaki (float) kalır
    data = pd.concat([matches.drop(columns=[c for c in feats.columns if c in matches.columns])
                      .reset_index(drop=True), feats], axis=1)
    # Eğitim satırları: 1. ligler + UEFA (2. ligler yalnızca geçmiş özellikleri besler)
    data = data[(data["date"] >= pd.Timestamp(dc.TRAIN_FIRST_SEASON_START, 7, 1))
                & ((data["tier"] == 1) | (data["is_uefa"] == 1.0))]
    before = len(data)
    data = data.dropna(subset=["elo_diff"])
    if len(data) < before:
        log.info("%d maç atıldı: takımlardan birinin geçmiş Elo'su yok (kulübün veri setindeki ilk maçı "
                 "ya da %d günden uzun ara)", before - len(data), ft.ELO_MAX_STALENESS_DAYS)
    data["target"] = data["result"].map(ft.RESULT_TO_CLASS).astype(int)
    data = data.sort_values("date", kind="stable").reset_index(drop=True)

    ctx_small = {k: ctx[k] for k in ("elo_tuning", "identity_report")}
    dc.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"signature": signature, "data": data, "store": store, "ctx": ctx_small}, DATASET_CACHE_PATH)
    return data, store, ctx_small


def candidate_groups(train_part: pd.DataFrame) -> tuple[dict[str, list[str]], list[dict]]:
    """Eğitim döneminde tamamen boş ya da sabit olan özellikler aday gruplardan çıkarılır (gerekçesiyle)."""
    groups, excluded = {}, []
    for group, cols in ft.FEATURE_GROUPS.items():
        kept = []
        for col in cols:
            s = train_part[col]
            if s.isna().all():
                excluded.append({"name": col, "reason": "eğitim verisinde tamamen boş"})
            elif s.nunique(dropna=True) <= 1:
                excluded.append({"name": col, "reason": f"eğitim verisinde sabit ({s.dropna().iloc[0]})"})
            else:
                kept.append(col)
        if kept:
            groups[group] = kept
    return groups, excluded


def subsets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    uefa = frame["is_uefa"].to_numpy(float) == 1.0
    return {
        "tum_maclar": np.ones(len(frame), dtype=bool),
        "ic_lig": ~uefa,
        "uefa_tumu": uefa,
        "uefa_ana_turnuva": frame["competition"].isin(UEFA_MAIN).to_numpy(),
        "uefa_eleme_turu": frame["is_knockout"].to_numpy(float) == 1.0,
    }


def market_reference(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Bahis oranlarından (marj normalize edilmiş) olasılıklar — sadece karşılaştırma ölçütü, özellik DEĞİL."""
    odds = frame[["odds_home", "odds_draw", "odds_away"]].apply(pd.to_numeric, errors="coerce")
    ok = (odds.notna().all(axis=1) & (odds > 1).all(axis=1)).to_numpy()
    if ok.sum() < 100:
        return None
    inv = 1.0 / odds[ok].to_numpy()
    return ok, inv / inv.sum(axis=1, keepdims=True)


# =============================================================================
# feature_schema.md
# =============================================================================

def library_versions() -> dict:
    import scipy
    return {"python": platform.python_version(), "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "joblib": joblib.__version__}


def write_feature_schema(meta: dict) -> None:
    contract = meta["contract"]
    ens = contract["ensemble"]
    lines = [
        "# feature_schema.md",
        "",
        "> **Bu dosya `src/train.py` tarafından otomatik üretilir — elle düzenlemeyin.**",
        "> `models/mac_modeli.pkl` ile birlikte taşınmalıdır. `predict.py`, en alttaki JSON bloğunu",
        "> okuyup modelin beklediği özelliklerle ve topluluk parametreleriyle birebir karşılaştırır.",
        "",
        f"- **Eğitim zamanı (UTC):** {meta['trained_at']}",
        f"- **Sözleşme hash'i:** `{meta['contract_hash']}`",
        "- **Model girdisi:** tek satırlık `pandas.DataFrame`, kolonlar aşağıdaki sırada, hepsi `float64`",
        f"- **Model çıktısı:** `[{', '.join(ft.CLASS_LABELS)}]` olasılıkları + (opsiyonel) skor matrisi",
        "",
        "## Model yapısı (topluluk)",
        "",
        "```text",
        "p_xgb     = components['xgb'].predict_proba(X)                  # XGBoost sınıflandırıcı",
        "p_logreg  = components['logreg'].predict_proba(X)               # Lojistik Regresyon pipeline (imputer+scaler)",
        "λ_ev      = components['poisson_home'].predict(X)               # XGBoost Poisson regresyonu (ev golü)",
        "λ_dep     = components['poisson_away'].predict(X)               # XGBoost Poisson regresyonu (deplasman golü)",
        f"M[i,j]    = Poisson(i; λ_ev) * Poisson(j; λ_dep), i,j = 0..{ens['max_goals']}",
        "            Dixon-Coles: M[0,0]*=1-λ_ev*λ_dep*ρ; M[0,1]*=1+λ_ev*ρ; M[1,0]*=1+λ_dep*ρ; M[1,1]*=1-ρ",
        "            negatifler 0'a kırpılır, M toplamı 1'e normalize edilir",
        "p_poisson = [Σ_{i>j} M, Σ_{i=j} M, Σ_{i<j} M]",
        "p         = w_xgb*p_xgb + w_logreg*p_logreg + w_poisson*p_poisson ; p /= Σp",
        "p_final   = softmax(log(max(p, 1e-6)) / T)",
        "skor dağılımı (opsiyonel): M'nin ev/beraberlik/deplasman bölgeleri p_final'a eşit toplamlara ölçeklenir",
        "```",
        "",
        f"- Ağırlıklar: `{json.dumps(ens['weights'])}`, Dixon-Coles ρ = `{ens['rho']:.5f}`, "
        f"sıcaklık T = `{ens['temperature']:.5f}`",
        f"- Bileşen hiperparametreleri: `{json.dumps(meta['component_params'])}`",
        "- Formülün referans uygulaması: `src/predict.py` (`component_outputs`, `combine`, `consistent_score_matrix`).",
        "",
        "## Veri kaynakları ve aralık",
        "",
        "- football-data.co.uk: 21 Avrupa 1. ligi + 6 adet 2. lig (maç sonuçları, kapanış oranları, şutlar)",
        "- Transfermarkt veri seti (dcaribou/transfermarkt-datasets, CC0): UEFA CL/EL/UECL + ön eleme "
        "turları, UKR/CRO/CZE/SRB ligleri, oyuncu maç kadroları, piyasa değerleri, oyuncu mevkileri",
        f"- Kaynakların son maç tarihleri: {meta['last_match_by_source']}",
        f"- Elo ısınma dönemi: {ft.FEATURE_PARAMS['ELO_DATA_START']} - {ft.FEATURE_PARAMS['ELO_BURN_IN_END']}",
        "- Eğitim satırları: 1. lig + UEFA maçları (2. lig maçları yalnızca geçmiş özellikleri besler)",
        "",
        "| Bölüm | Başlangıç | Bitiş | Maç sayısı | UEFA maçı |",
        "|---|---|---|---|---|",
    ]
    for part, r in meta["data_range"].items():
        lines.append(f"| {part} | {r['from']} | {r['to']} | {r['n']} | {r['n_uefa']} |")
    lines += [
        "",
        f"Çapraz doğrulama sezonları: {meta['cv_folds']}. Kaydedilen model şu veriyle eğitildi: **{meta['fitted_on']}**.",
        "",
        "## Özellikler (sıra önemlidir)",
        "",
        "| # | Ad | Grup | Tip | Kaynak | Eğitimde NaN oranı |",
        "|---|---|---|---|---|---|",
    ]
    for i, spec in enumerate(contract["features"], 1):
        lines.append(f"| {i} | `{spec['name']}` | {spec['group']} | {spec['dtype']} | {spec['source']} | "
                     f"{meta['nan_rates'][spec['name']]:.1%} |")
    lines += ["", "### Hesaplama tanımları", ""]
    for spec in contract["features"]:
        lines += [f"**`{spec['name']}`**", "", "```text", spec["definition"], "```", ""]
    sel = meta["selection"]
    lines += [
        "## Özellik grubu seçimi (kayan çapraz doğrulama)",
        "",
        "İleri seçim: `base` grubundan başlanır; her turda CV ölçütünü (0.5 × log-loss tüm maçlar + 0.5 × "
        f"log-loss UEFA) en çok iyileştiren grup eklenir; iyileşme {SELECTION_MIN_IMPROVEMENT}'ten küçükse durulur. "
        f"Test seti kullanılmaz. Seçilen gruplar: **{', '.join(meta['chosen_groups'])}**.",
        "",
        "| Tur | Denenen grup | Ölçüt | Log-loss (tüm) | Log-loss (UEFA) | Eklendi |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r.round} | {r.group} | {r.objective:.5f} | {r.cv_log_loss:.5f} | "
              f"{r.cv_uefa_log_loss:.5f} | {'✔' if r.added else ''} |" for r in sel.itertuples()]
    lines += ["", "## Modelden çıkarılan özellikler", ""]
    lines += [f"- `{e['name']}`: {e['reason']}" for e in meta["excluded"]] or ["- (yok)"]
    lines += [
        "",
        "## Eğitimde ayarlanan Elo parametreleri",
        "",
        f"Izgara araması yalnızca {meta['elo_tune_until']} öncesi maçlarla yapıldı (Elo beklenen skorunun "
        "ortalama kare hatası; iç lig ve UEFA eşit ağırlıklı). Tam tablo: `models/elo_tuning.csv`.",
        "",
        "```json", json.dumps(contract["elo_params"], indent=2), "```",
        "",
        "## Sabit parametreler",
        "",
        "```json", json.dumps(contract["params"], indent=2, ensure_ascii=False), "```",
        "",
        "## Eksik değer politikası",
        "",
        "- `elo_diff` NaN ise tahmin YAPILMAMALI (eğitimde bu satırlar atıldı).",
        "- Diğer özellikler NaN olabilir: XGBoost bileşenleri NaN'ı kendi içinde işler; Lojistik Regresyon",
        "  pipeline'ı eğitim medyanıyla doldurur. Doldurma işi pkl'ın içindedir, kullanan tarafta ayrıca",
        "  doldurma YAPILMAMALI (NaN olduğu gibi verilmeli).",
        "",
        "## Test metrikleri",
        "",
        "| Alt küme | Model | n | Accuracy | Log-loss | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for subset, models in meta["test_metrics"].items():
        for name, m in models.items():
            lines.append(f"| {subset} | {name} | {m['n']} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} |")
    lines += [
        "",
        "`ensemble` = kaydedilen modelin çıktısı. `market_odds_reference`: kapanış oranlarından türetilen olasılıklar "
        "(yalnızca iç lig; kıyas ölçütüdür, modelde kullanılmaz).",
        "",
        "Kalibrasyon (ensemble, test; ECE = beklenen kalibrasyon hatası, 0 ideal):",
        "",
    ]
    for subset, e in meta["ece"].items():
        lines.append(f"- {subset}: " + ", ".join(f"{k}={v:.4f}" for k, v in e.items()))
    lines += ["", "## Kütüphane versiyonları", "", "| Paket | Versiyon |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in meta["library_versions"].items()]
    machine = {
        "schema_version": 3,
        "contract_hash": meta["contract_hash"],
        **contract,
        "component_params": meta["component_params"],
        "excluded_features": meta["excluded"],
        "trained_at": meta["trained_at"],
        "data_range": meta["data_range"],
        "fitted_on": meta["fitted_on"],
        "data_end": meta["data_end"],
        "library_versions": meta["library_versions"],
    }
    lines += ["", "## Makine tarafından okunan sözleşme", "", SCHEMA_JSON_BEGIN, "```json",
              json.dumps(machine, indent=2, ensure_ascii=False), "```", SCHEMA_JSON_END, ""]
    SCHEMA_PATH.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Ana akış
# =============================================================================

def _range(df: pd.DataFrame) -> dict:
    return {"from": str(df["date"].min().date()), "to": str(df["date"].max().date()),
            "n": int(len(df)), "n_uefa": int((df["is_uefa"] == 1.0).sum())}


def set_output_dir(out_dir) -> None:
    """
    Tüm çıktıları (pkl, feature_schema.md, metrikler) başka bir klasöre yazdırır. Haftalık güncelleme,
    yeni modeli önce bu şekilde ayrı bir klasörde eğitir; kalite kontrolünü geçerse üretime alır.
    """
    global MODELS_DIR, MODEL_PATH, SCHEMA_PATH, METRICS_PATH, CALIBRATION_PATH, ELO_TUNING_PATH
    global FEATURE_SELECTION_PATH, HYPERPARAM_PATH
    from pathlib import Path
    MODELS_DIR = Path(out_dir)
    MODEL_PATH = MODELS_DIR / "mac_modeli.pkl"
    SCHEMA_PATH = MODELS_DIR / "feature_schema.md"
    METRICS_PATH = MODELS_DIR / "metrics.json"
    CALIBRATION_PATH = MODELS_DIR / "calibration_report.csv"
    ELO_TUNING_PATH = MODELS_DIR / "elo_tuning.csv"
    FEATURE_SELECTION_PATH = MODELS_DIR / "feature_selection.csv"
    HYPERPARAM_PATH = MODELS_DIR / "hyperparameter_search.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Maç sonucu modeli eğitimi")
    parser.add_argument("--test-from", type=pd.Timestamp, default=None)
    parser.add_argument("--reuse-dataset", action="store_true", help="Özellik tablosunu önbellekten al")
    parser.add_argument("--refit-all", action="store_true",
                        help="Testten sonra geliştirme + test verisiyle yeniden eğitip onu kaydet")
    parser.add_argument("--output-dir", default=None,
                        help="Çıktıları models/ yerine bu klasöre yaz (feature_schema.md dahil)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.output_dir:
        set_output_dir(args.output_dir)

    test_from = args.test_from or default_test_from(date.today())
    folds = make_folds(test_from)
    elo_tune_until = folds[0][0]
    data, store, ctx = prepare_dataset(elo_tune_until, reuse=args.reuse_dataset)
    dev, test = data[data["date"] < test_from], data[data["date"] >= test_from]
    cv = RollingCV(dev, folds)
    log.info("Geliştirme %s -> %s (%d maç), test %s -> %s (%d maç, %d UEFA)", dev["date"].min().date(),
             dev["date"].max().date(), len(dev), test["date"].min().date(), test["date"].max().date(),
             len(test), int((test["is_uefa"] == 1.0).sum()))
    log.info("CV sezonları: %s", [str(a.date()) for a, _ in folds])

    # a) özellik grupları
    groups, excluded = candidate_groups(dev[dev["date"] < elo_tune_until])
    for e in excluded:
        log.warning("Özellik aday listesinden çıkarıldı: %s — %s", e["name"], e["reason"])
    log.info("a) Özellik grubu ileri seçimi (CV):")
    cols, selection = select_feature_groups(cv, groups)
    chosen_groups = selection.loc[selection["added"], "group"].tolist()
    excluded += [{"name": c, "reason": f"'{g}' grubu CV ileri seçiminde modele eklenmedi"}
                 for g, gcols in groups.items() if g not in chosen_groups for c in gcols]
    log.info("Seçilen gruplar: %s; özellikler (%d): %s", chosen_groups, len(cols), cols)

    # b) XGBoost sınıflandırıcı
    log.info("b) XGBoost hiperparametre araması (CV):")
    xgb_params, hp_table, oof_xgb = search_xgb_classifier(cv, cols)
    log.info("  en iyi: %s", xgb_params)

    # c) Lojistik Regresyon
    log.info("c) Lojistik Regresyon (CV):")
    best_lr = None
    for C in (0.01, 0.1, 1.0):
        for hl in (None, 4.0):
            oof = cv.logreg(cols, C, hl)
            obj = cv.score(oof)[0]
            log.info("  C=%s half_life=%s -> ölçüt=%.5f", C, hl, obj)
            if best_lr is None or obj < best_lr[0]:
                best_lr = (obj, {"C": C, "half_life": hl}, oof)
    lr_params, oof_lr = best_lr[1], best_lr[2]

    # d) Poisson skor modelleri + Dixon-Coles rho
    log.info("d) Poisson skor modelleri (CV):")
    best_po = None
    for depth in (3, 4):
        p = {"learning_rate": 0.03, "max_depth": depth, "min_child_weight": 50,
             "colsample_bytree": xgb_params["colsample_bytree"], "reg_lambda": xgb_params["reg_lambda"],
             "half_life": xgb_params["half_life"]}
        lam_h, lam_a, iters = cv.poisson(cols, p)
        rho = fit_rho(lam_h, lam_a, cv.goals_h, cv.goals_a)
        wdl = wdl_from_matrix(dixon_coles_matrix(lam_h, lam_a, rho))
        obj = cv.score(wdl)[0]
        log.info("  depth=%d rho=%.4f -> ölçüt=%.5f", depth, rho, obj)
        if best_po is None or obj < best_po[0]:
            best_po = (obj, {**p, "n_estimators": int(np.median(iters) * 1.1)}, rho, wdl)
    po_params, rho, oof_po = best_po[1], best_po[2], best_po[3]

    # e) topluluk ağırlıkları + sıcaklık
    log.info("e) Topluluk ağırlıkları ve sıcaklık (CV):")
    oof = {"xgb": oof_xgb, "logreg": oof_lr, "poisson": oof_po}
    for name, p in oof.items():
        log.info("  tek başına %-8s ölçüt=%.5f", name, cv.score(p)[0])
    best_w = None
    steps = np.round(np.arange(0, 1.0001, 0.05), 2)
    for w1 in steps:
        for w2 in steps:
            if w1 + w2 > 1.0001:
                continue
            w = {"xgb": float(w1), "logreg": float(w2), "poisson": float(round(1 - w1 - w2, 2))}
            obj = cv.score(combine(oof, w, 1.0))[0]
            if best_w is None or obj < best_w[0]:
                best_w = (obj, w)
    weights = best_w[1]
    temp = float(minimize_scalar(lambda t: cv.score(combine(oof, weights, t))[0], bounds=(0.7, 1.5),
                                 method="bounded").x)
    cv_final = cv.score(combine(oof, weights, temp))
    log.info("  ağırlıklar=%s T=%.4f -> CV ölçüt=%.5f (tüm %.5f, UEFA %.5f)", weights, temp, *cv_final)
    ensemble = {"weights": weights, "rho": rho, "temperature": temp, "max_goals": MAX_GOALS}

    # 4) son modeller: tüm geliştirme verisi
    def fit_components(frame: pd.DataFrame, ref: pd.Timestamp) -> dict:
        comps = {"xgb": make_xgb_classifier(xgb_params, xgb_params["n_estimators"], early_stopping=False).fit(
            frame[cols], frame["target"], sample_weight=time_weights(frame["date"], ref, xgb_params["half_life"]),
            verbose=False)}
        comps["logreg"] = make_logreg(lr_params["C"]).fit(
            frame[cols], frame["target"], clf__sample_weight=time_weights(frame["date"], ref, lr_params["half_life"]))
        w = time_weights(frame["date"], ref, po_params["half_life"])
        for target, name in (("home_goals", "poisson_home"), ("away_goals", "poisson_away")):
            comps[name] = make_xgb_poisson(po_params, po_params["n_estimators"], early_stopping=False).fit(
                frame[cols], frame[target], sample_weight=w, verbose=False)
        return comps

    log.info("Son modeller geliştirme verisiyle eğitiliyor...")
    components = fit_components(dev, test_from)
    X_te, y_te = test[cols], test["target"].to_numpy()
    outputs = component_outputs(components, X_te, rho)
    test_proba = {
        "naive_class_frequencies": np.tile(np.bincount(dev["target"], minlength=3) / len(dev), (len(test), 1)),
        "xgb": outputs["xgb"], "logreg": outputs["logreg"], "poisson": outputs["poisson"],
        "ensemble": combine(outputs, weights, temp),
    }
    test_metrics: dict[str, dict] = {}
    for subset, mask in subsets(test).items():
        if mask.sum() == 0:
            continue
        test_metrics[subset] = {name: evaluate(y_te[mask], p[mask]) for name, p in test_proba.items()}
        market = market_reference(test[mask])
        if market is not None:
            ok, p_market = market
            test_metrics[subset]["market_odds_reference"] = evaluate(y_te[mask][ok], p_market)
            test_metrics[subset]["ensemble_on_market_subset"] = evaluate(y_te[mask][ok], test_proba["ensemble"][mask][ok])

    calib_frames, ece = [], {}
    for subset in ("tum_maclar", "uefa_tumu"):
        mask = subsets(test)[subset]
        table = calibration_table(y_te[mask], test_proba["ensemble"][mask]).assign(subset=subset)
        calib_frames.append(table)
        ece[subset] = expected_calibration_error(table)
    calib = pd.concat(calib_frames, ignore_index=True)

    print("\n=== TEST METRİKLERİ ===")
    rows = [{"alt_kume": s, "model": m, **v} for s, ms in test_metrics.items() for m, v in ms.items()]
    print(pd.DataFrame(rows)[["alt_kume", "model", "n", "accuracy", "log_loss", "brier"]].to_string(index=False))
    for subset in ("tum_maclar", "uefa_tumu"):
        print(f"\n=== KALİBRASYON (ensemble, test, {subset}) — ECE: "
              + ", ".join(f"{k}={v:.4f}" for k, v in ece[subset].items()) + " ===")
        view = calib[(calib["subset"] == subset) & (calib["target"] == "top_label")]
        print(view.drop(columns=["subset"]).to_string(index=False))

    fitted_on = f"geliştirme verisi ({dev['date'].min().date()} - {dev['date'].max().date()})"
    data_end = dev["date"].max()
    if args.refit_all:
        components = fit_components(data, data["date"].max() + pd.Timedelta(days=1))
        fitted_on = "geliştirme + test (--refit-all; test metrikleri bir önceki fit'e ait)"
        data_end = data["date"].max()

    contract = ft.schema_contract(cols, store.elo_params, ensemble)
    chash = ft.contract_hash(contract)
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    component_params = {"xgb": xgb_params, "logreg": lr_params, "poisson": po_params}
    meta = {
        "contract": contract, "contract_hash": chash, "trained_at": trained_at,
        "component_params": component_params,
        "data_range": {"development": _range(dev), "test": _range(test)},
        "cv_folds": [f"{a.date()} - {b.date()}" for a, b in folds], "elo_tune_until": str(elo_tune_until.date()),
        "fitted_on": fitted_on, "data_end": str(data_end.date()),
        "excluded": excluded, "selection": selection, "chosen_groups": chosen_groups,
        "nan_rates": {c: float(dev[c].isna().mean()) for c in cols},
        "test_metrics": test_metrics, "ece": ece, "library_versions": library_versions(),
        "last_match_by_source": {k: str(v.date()) for k, v in store.last_match_by_source.items()},
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "components": components,
        "ensemble": contract["ensemble"],
        "component_params": component_params,
        "feature_columns": cols,
        "class_labels": ft.CLASS_LABELS,
        "elo_params": store.elo_params,
        "contract_hash": chash,
        "trained_at": trained_at,
        "data_end": meta["data_end"],
        "library_versions": meta["library_versions"],
    }, MODEL_PATH)
    write_feature_schema(meta)
    calib.to_csv(CALIBRATION_PATH, index=False)
    selection.to_csv(FEATURE_SELECTION_PATH, index=False)
    hp_table.to_csv(HYPERPARAM_PATH, index=False)
    if ctx["elo_tuning"] is not None:
        ctx["elo_tuning"].to_csv(ELO_TUNING_PATH, index=False)
    METRICS_PATH.write_text(json.dumps({
        "elo_params": store.elo_params, "chosen_groups": chosen_groups, "component_params": component_params,
        "ensemble": contract["ensemble"], "cv_objective": {"objective": cv_final[0], "log_loss": cv_final[1],
                                                           "uefa_log_loss": cv_final[2]},
        "test_metrics": test_metrics, "ece": ece, "data_range": meta["data_range"], "trained_at": trained_at,
        "data_end": meta["data_end"], "fitted_on": fitted_on,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # predict.py'nin özellikleri yeniden hesaplamaması için aynı veri + aynı Elo parametreleriyle önbellek
    store.save()
    log.info("Kaydedildi: %s, %s", MODEL_PATH, SCHEMA_PATH)


if __name__ == "__main__":
    main()
