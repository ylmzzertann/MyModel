"""
Tahmin arayüzü — ileride başka projeye taşınacak parça.

    from src.predict import MatchPredictor

    predictor = MatchPredictor()   # models/mac_modeli.pkl + feature_schema.md yükler
    probs = predictor.predict(home_team="Galatasaray", away_team="Bayern Munich", date="2026-10-21")
    # -> {"home_win": ..., "draw": ..., "away_win": ...}
    score = predictor.predict_score("Galatasaray", "Bayern Munich", "2026-10-21")
    # -> beklenen goller, skor olasılık matrisi, en olası skorlar

Model bir topluluktur (ensemble):
    p_xgb     = XGBoost sınıflandırıcı olasılıkları
    p_logreg  = Lojistik Regresyon olasılıkları
    p_poisson = iki Poisson regresyonunun (ev / deplasman golü) Dixon-Coles skor matrisinden G/B/M
    p         = w_xgb * p_xgb + w_logreg * p_logreg + w_poisson * p_poisson
    p_final   = softmax(log(p) / T)            (sıcaklık kalibrasyonu)
Eğitim (train.py) bu dosyadaki fonksiyonları kullanır; formül iki tarafta da birebir aynıdır.

Güvenlik kontrolleri (hepsi AÇIK HATA verir, sessizce devam etmez):
  1. pkl içindeki özellik listesi == feature_schema.md'deki liste (ad + sıra)
  2. feature_schema.md'deki sözleşme hash'i == pkl'daki hash
  3. Bu koddaki özellik tanımları/parametreleri == eğitimdekiler
  4. Her tahminde hesaplanan kolonlar şemadaki ad, sıra ve tiple birebir aynı
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import data_collection as dc
from . import features as ft

MODEL_PATH = dc.PROJECT_ROOT / "models" / "mac_modeli.pkl"
SCHEMA_PATH = dc.PROJECT_ROOT / "feature_schema.md"

_SCHEMA_BLOCK = re.compile(
    r"<!-- SCHEMA_JSON_BEGIN -->\s*```json\s*(\{.*?\})\s*```\s*<!-- SCHEMA_JSON_END -->", re.S)

# Tahmin tarihi, eldeki en güncel maç verisinden bu kadar gün sonrasıysa uyarı verilir
STALE_DATA_WARNING_DAYS = 45
MAX_GOALS = 10                     # skor matrisi 0..10 gol
PROBA_EPS = 1e-6


class SchemaMismatchError(RuntimeError):
    """Model, şema ve özellik hesaplaması birbiriyle uyuşmuyor."""


class UnknownTeamError(ValueError):
    """Takım adı veri setindeki bir kulübe çözümlenemedi."""


# =============================================================================
# Topluluk formülü (train.py de bunları kullanır)
# =============================================================================

def dixon_coles_matrix(lam_home: np.ndarray, lam_away: np.ndarray, rho: float,
                       max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    (n, G+1, G+1) skor olasılık matrisi; [i, j] = ev i gol, deplasman j gol.
    Bağımsız Poisson x Dixon-Coles düşük skor düzeltmesi:
      (0,0) *= 1 - λ_ev λ_dep ρ ; (0,1) *= 1 + λ_ev ρ ; (1,0) *= 1 + λ_dep ρ ; (1,1) *= 1 - ρ
    Negatif hücreler 0'a kırpılır, matris 1'e normalize edilir (0..G dışındaki kütle dahil edilmez).
    """
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], lam_home[:, None])
    pa = poisson.pmf(goals[None, :], lam_away[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1.0 - lam_home * lam_away * rho
    m[:, 0, 1] *= 1.0 + lam_home * rho
    m[:, 1, 0] *= 1.0 + lam_away * rho
    m[:, 1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)


def wdl_from_matrix(m: np.ndarray) -> np.ndarray:
    """Skor matrisinden [ev kazanır, beraberlik, deplasman kazanır]."""
    home = np.tril(m, k=-1).sum(axis=(1, 2))
    draw = np.trace(m, axis1=1, axis2=2)
    away = np.triu(m, k=1).sum(axis=(1, 2))
    return np.stack([home, draw, away], axis=1)


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(p, PROBA_EPS, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def component_outputs(components: dict, X: pd.DataFrame, rho: float) -> dict:
    """Her bileşenin G/B/M olasılıkları + Poisson beklenen golleri ve skor matrisi."""
    lam_h = components["poisson_home"].predict(X)
    lam_a = components["poisson_away"].predict(X)
    matrix = dixon_coles_matrix(lam_h, lam_a, rho)
    return {
        "xgb": components["xgb"].predict_proba(X),
        "logreg": components["logreg"].predict_proba(X),
        "poisson": wdl_from_matrix(matrix),
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "score_matrix": matrix,
    }


def combine(outputs: dict, weights: dict, temperature: float) -> np.ndarray:
    p = sum(weights[name] * outputs[name] for name in ("xgb", "logreg", "poisson"))
    return apply_temperature(p / p.sum(axis=1, keepdims=True), temperature)


def consistent_score_matrix(matrix: np.ndarray, final_wdl: np.ndarray) -> np.ndarray:
    """
    Skor matrisini, topluluğun nihai G/B/M olasılıklarıyla tutarlı hale getirir: ev kazanır / beraberlik /
    deplasman kazanır bölgelerindeki hücreler, bölge toplamı nihai olasılığa eşit olacak şekilde ölçeklenir.
    Bölge içindeki skorların göreli dağılımı (Poisson + Dixon-Coles) korunur.
    """
    base = wdl_from_matrix(matrix)
    g = matrix.shape[1]
    i, j = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    region = np.where(i > j, 0, np.where(i == j, 1, 2))
    scale = final_wdl / np.clip(base, PROBA_EPS, None)
    return matrix * scale[:, region]


# =============================================================================
# Şema
# =============================================================================

def load_schema(path: Path = SCHEMA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"feature_schema.md bulunamadı: {path} — model şemasız kullanılamaz.")
    match = _SCHEMA_BLOCK.search(path.read_text(encoding="utf-8"))
    if not match:
        raise SchemaMismatchError(f"{path.name} içinde makine okunur JSON bloğu bulunamadı.")
    return json.loads(match.group(1))


class MatchPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, schema_path: Path = SCHEMA_PATH,
                 store: ft.FeatureStore | None = None):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model dosyası yok: {model_path} — önce `python -m src.train`.")
        self.bundle = joblib.load(model_path)
        self.schema = load_schema(Path(schema_path))
        self.components = self.bundle["components"]
        self.feature_columns: list[str] = list(self.schema["feature_columns"])
        self._validate_contract()
        self._check_library_versions()
        self.store = store if store is not None else ft.FeatureStore.load_or_build(self.schema["elo_params"])
        if {k: float(v) for k, v in self.store.elo_params.items()} != self.schema["elo_params"]:
            raise SchemaMismatchError(
                f"Özellik deposu farklı Elo parametreleriyle kurulmuş: {self.store.elo_params} "
                f"!= şema {self.schema['elo_params']}")

    # ------------------------------------------------------------------ doğrulama
    def _validate_contract(self) -> None:
        bundle_cols = list(self.bundle["feature_columns"])
        if bundle_cols != self.feature_columns:
            raise SchemaMismatchError(
                f"pkl özellikleri {bundle_cols} != feature_schema.md özellikleri {self.feature_columns}")

        keys = ("feature_columns", "features", "params", "elo_params", "class_order", "ensemble")
        schema_contract = {k: self.schema[k] for k in keys}
        if ft.contract_hash(schema_contract) != self.schema["contract_hash"]:
            raise SchemaMismatchError("feature_schema.md içeriği hash'iyle uyuşmuyor (elle değiştirilmiş olabilir).")
        if self.schema["contract_hash"] != self.bundle["contract_hash"]:
            raise SchemaMismatchError("feature_schema.md bu pkl dosyasına ait değil (hash farklı).")

        code_contract = ft.schema_contract(self.feature_columns, self.schema["elo_params"], self.schema["ensemble"])
        if ft.contract_hash(code_contract) != self.schema["contract_hash"]:
            diffs = [k for k in ("features", "params", "class_order") if code_contract[k] != schema_contract[k]]
            raise SchemaMismatchError(
                f"src/features.py eğitimdeki tanımlardan farklı ({', '.join(diffs)}). "
                "Modeli yeniden eğitin ya da eğitimdeki features.py sürümüne dönün.")

        ens = self.schema["ensemble"]
        if (ens["weights"] != self.bundle["ensemble"]["weights"] or ens["rho"] != self.bundle["ensemble"]["rho"]
                or ens["temperature"] != self.bundle["ensemble"]["temperature"]):
            raise SchemaMismatchError("Şemadaki topluluk parametreleri pkl ile uyuşmuyor.")
        classes = [list(getattr(self.components[n], "classes_", [0, 1, 2])) for n in ("xgb", "logreg")]
        if any(c != [0, 1, 2] for c in classes) or self.schema["class_order"] != ft.CLASS_LABELS:
            raise SchemaMismatchError(f"Sınıf sırası beklenenden farklı: {classes} / {self.schema['class_order']}")

    def _check_library_versions(self) -> None:
        import sklearn
        import xgboost
        installed = {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__}
        trained = self.schema.get("library_versions", {})
        for pkg, ver in installed.items():
            if trained.get(pkg) and trained[pkg] != ver:
                warnings.warn(f"{pkg} versiyonu farklı: eğitim {trained[pkg]}, kurulu {ver}. "
                              "pkl yüklenmiş olsa bile tahminler değişebilir.", stacklevel=3)

    def _validate_features(self, X: pd.DataFrame, n_rows: int = 1) -> None:
        # `assert` yerine açık raise: `python -O` ile çalıştırılınca assert'ler
        # tamamen devre dışı kalır, bu kontrollerin ise asla atlanmaması gerekir.
        checks = [
            (list(X.columns) == self.feature_columns,
             f"Hesaplanan özellikler {list(X.columns)} != şema {self.feature_columns}"),
            (X.shape == (n_rows, len(self.feature_columns)), f"Beklenmeyen girdi boyutu: {X.shape}"),
            (all(dt == np.float64 for dt in X.dtypes), f"Tüm özellikler float64 olmalı: {X.dtypes.to_dict()}"),
            (not np.isinf(X.to_numpy()).any(), "Özelliklerde sonsuz değer var"),
        ]
        for ok, message in checks:
            if not ok:
                raise SchemaMismatchError(message)

    # ------------------------------------------------------------------ özellikler
    def resolve_team(self, name: str) -> str:
        """Takım adını kanonik anahtara çevirir ('tm:<id>' ya da 'fd:<ülke>:<ad>')."""
        key, info = self.store.identity.resolve(name)
        if key is None:
            raise UnknownTeamError(
                f"'{name}' tek bir kulübe çözümlenemedi. Adaylar: {info}. Anahtarı doğrudan verebilir "
                "(ör. 'tm:141') ya da data/team_aliases.csv dosyasına 'alias,target' satırı ekleyebilirsiniz.")
        return key

    def infer_is_uefa(self, home_key: str, away_key: str, is_knockout: bool) -> bool:
        """
        is_uefa verilmediğinde: eleme turu ise ya da kulüplerin ülkeleri farklı / bilinmiyorsa UEFA,
        aynı ülkenin iki kulübüyse iç lig maçı kabul edilir. (Aynı ülkeden iki kulübün UEFA'da
        karşılaştığı nadir durumda is_uefa=True açıkça verilmeli.)
        """
        if is_knockout:
            return True
        c_home = self.store.identity.country.get(home_key)
        c_away = self.store.identity.country.get(away_key)
        return c_home is None or c_away is None or c_home != c_away

    def features_for(self, home_team: str, away_team: str, date, neutral: bool = False,
                     is_knockout: bool = False, is_uefa: bool | None = None) -> pd.DataFrame:
        """Modelin göreceği özellik satırını döndürür (teşhis için de kullanılabilir)."""
        home_key, away_key = self.resolve_team(home_team), self.resolve_team(away_team)
        if is_uefa is None:
            is_uefa = self.infer_is_uefa(home_key, away_key, is_knockout)
        fx = pd.DataFrame([{
            "date": pd.Timestamp(date),
            "home_key": home_key,
            "away_key": away_key,
            "neutral": float(neutral),
            "is_knockout": float(is_knockout),
            "is_uefa": float(is_uefa),
        }])
        feats = self.store.compute(fx)
        feats.insert(0, "home_club", self.store.identity.display[home_key])
        feats.insert(1, "away_club", self.store.identity.display[away_key])
        return feats

    def _model_input(self, home_team, away_team, date, neutral, is_knockout, is_uefa, overrides) -> pd.DataFrame:
        feats = self.features_for(home_team, away_team, date, neutral, is_knockout, is_uefa)
        for name, value in (overrides or {}).items():
            if name not in self.feature_columns:
                raise SchemaMismatchError(f"overrides: '{name}' modelin özelliklerinden biri değil "
                                          f"({self.feature_columns})")
            if name in ("elo_diff", "mkt_elo_diff", "is_knockout", "is_uefa"):
                raise SchemaMismatchError(f"overrides: '{name}' parametrelerle / veriden belirlenir, dışarıdan verilemez")
            feats.loc[0, name] = float(value)
        if overrides and "xi_value_diff" in self.feature_columns and "xi_value_diff" not in overrides \
                and {"home_xi_value", "away_xi_value"} & set(overrides):
            # türetilmiş özellik, bileşenleri değişince tutarlı kalmalı
            feats.loc[0, "xi_value_diff"] = feats.loc[0, "home_xi_value"] - feats.loc[0, "away_xi_value"]

        if np.isnan(feats.loc[0, "elo_diff"]):
            missing = [feats.loc[0, f"{side}_club"] for side in ("home", "away")
                       if np.isnan(feats.loc[0, f"{side}_elo"])]
            raise ValueError(f"{date} için Elo bulunamadı: {missing} (son {ft.ELO_MAX_STALENESS_DAYS} "
                             "günde veri setinde maçı yok). Elo olmadan tahmin yapılmaz.")
        when = pd.Timestamp(date)
        if (when - self.store.last_match_date).days > STALE_DATA_WARNING_DAYS:
            warnings.warn(f"En güncel maç verisi {self.store.last_match_date.date()}; tahmin tarihi "
                          f"{when.date()}. Veriyi güncellemeyi düşünün.", stacklevel=3)
        nan_used = [c for c in self.feature_columns if np.isnan(feats.loc[0, c])]
        if nan_used:
            warnings.warn(f"Eksik (NaN) özellik: {nan_used} — model bunları eğitimdeki gibi işler, "
                          "ama tahmin daha az bilgiye dayanır.", stacklevel=3)
        X = feats[self.feature_columns]
        self._validate_features(X)
        return X

    # ------------------------------------------------------------------ tahmin
    def _outputs(self, X: pd.DataFrame) -> tuple[dict, np.ndarray]:
        ens = self.bundle["ensemble"]
        outputs = component_outputs(self.components, X, ens["rho"])
        return outputs, combine(outputs, ens["weights"], ens["temperature"])

    def predict(self, home_team: str, away_team: str, date, neutral: bool = False,
                is_knockout: bool = False, is_uefa: bool | None = None,
                overrides: dict[str, float] | None = None) -> dict[str, float]:
        """
        neutral: tarafsız saha (ör. final) -> ev avantajı sıfırlanır
        is_knockout: UEFA grup/lig aşaması sonrası eleme turu
        is_uefa: UEFA maçı mı; None ise infer_is_uefa() kuralıyla çıkarılır
        overrides: veriden hesaplanan bir özelliği dışarıdan verilen değerle değiştirir
            (ör. simülasyon kendi kadro verisinden {"home_xi_value": ...} verebilir). Yalnızca şemadaki
            özellik adları kabul edilir; değer feature_schema.md'deki tanımla AYNI şekilde hesaplanmalıdır.
        """
        X = self._model_input(home_team, away_team, date, neutral, is_knockout, is_uefa, overrides)
        _, final = self._outputs(X)
        return {label: float(p) for label, p in zip(ft.CLASS_LABELS, final[0])}

    def predict_many(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Toplu tahmin (ör. bir turnuvadaki tüm eşleşmeler). predict() ile AYNI hesaplama ve doğrulama.
        fixtures kolonları: date, home_key, away_key, neutral, is_knockout, is_uefa (anahtarlar resolve_team ile).
        Dönüş: her satır için home_win, draw, away_win, lambda_home, lambda_away, elo_missing ve özellikler.
        elo_missing olan satırlarda olasılıklar NaN'dır (Elo olmadan tahmin yapılmaz).
        """
        fx = fixtures.reset_index(drop=True)
        feats = self.store.compute(fx[["date", "home_key", "away_key", "neutral", "is_knockout", "is_uefa"]])
        elo_missing = feats["elo_diff"].isna().to_numpy()
        X = feats[self.feature_columns]
        self._validate_features(X, n_rows=len(fx))
        out = pd.DataFrame({"home_win": np.nan, "draw": np.nan, "away_win": np.nan,
                            "lambda_home": np.nan, "lambda_away": np.nan, "elo_missing": elo_missing})
        ok = ~elo_missing
        if ok.any():
            outputs, final = self._outputs(X[ok])
            out.loc[ok, ["home_win", "draw", "away_win"]] = final
            out.loc[ok, "lambda_home"] = outputs["lambda_home"]
            out.loc[ok, "lambda_away"] = outputs["lambda_away"]
        return pd.concat([out, feats[self.feature_columns]], axis=1)

    def predict_score(self, home_team: str, away_team: str, date, neutral: bool = False,
                      is_knockout: bool = False, is_uefa: bool | None = None,
                      overrides: dict[str, float] | None = None, top_n: int = 10) -> dict:
        """
        Skor dağılımı (simülasyon için):
          expected_goals   : Poisson modellerinin beklenen golleri (λ_ev, λ_dep)
          score_matrix     : (11x11) olasılıklar, [i][j] = ev i - deplasman j; G/B/M bölge toplamları
                             predict() çıktısına eşit olacak şekilde ölçeklenmiş (consistent_score_matrix)
          top_scores       : en olası top_n skor
          wdl              : predict() ile aynı olasılıklar
        Uzatma için (30 dk) yaklaşık λ * 30/90 kullanılabilir; bu model uzatmalarla eğitilmedi.
        """
        X = self._model_input(home_team, away_team, date, neutral, is_knockout, is_uefa, overrides)
        outputs, final = self._outputs(X)
        matrix = consistent_score_matrix(outputs["score_matrix"], final)[0]
        order = np.dstack(np.unravel_index(np.argsort(-matrix, axis=None), matrix.shape))[0][:top_n]
        return {
            "expected_goals": {"home": float(outputs["lambda_home"][0]), "away": float(outputs["lambda_away"][0])},
            "score_matrix": matrix.round(6).tolist(),
            "top_scores": [{"score": f"{i}-{j}", "probability": float(matrix[i, j])} for i, j in order],
            "wdl": {label: float(p) for label, p in zip(ft.CLASS_LABELS, final[0])},
        }
