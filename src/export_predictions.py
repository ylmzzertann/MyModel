"""
Simülasyon için tahmin tablosu dışa aktarımı.

    python -m src.export_predictions --teams-dir "C:/.../Champions League Simulator/API/data" --date 2026-09-14

Simülasyonun takım listelerini ({turnuva}/{turnuva}_teams.json, "Pot" -> [{name, country}]) okur, her takımı
modelin kanonik kulüp anahtarına eşler ve her turnuvada tüm ev/deplasman eşleşmeleri için üç maç türünde
tahmin üretir:
    league   : lig aşaması              (is_uefa=1, is_knockout=0, neutral=0)
    knockout : eleme turu maçı          (is_uefa=1, is_knockout=1, neutral=0)
    final    : tek maçlık final         (is_uefa=1, is_knockout=1, neutral=1)

Çıktı (exports/sim_predictions_<tarih>.json), maç başına G/B/M olasılıkları ve iki beklenen gol (λ) içerir.
Skor olasılık matrisi bu değerlerden + dosyadaki rho'dan yeniden kurulur (bkz. ENTEGRASYON.md);
--full-matrix ile 11x11 matrisler de dosyaya yazılır.

Takım eşleştirmesi data/sim_team_mapping.csv dosyasında tutulur:
  - dosyada anahtarı dolu satırlar olduğu gibi kullanılır (elle doğrulanmış kabul edilir),
  - eksik takımlar ülke kontrollü otomatik eşleştirmeyle eklenir,
  - eşleşemeyen takım varsa aday listesiyle birlikte hata verilir; tahmin tablosu YAZILMAZ.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_collection as dc
from . import features as ft
from .predict import MAX_GOALS, MatchPredictor

log = logging.getLogger(__name__)

COMPETITIONS = ("ucl", "uel", "uecl")
STAGES = {
    "league": {"is_knockout": 0.0, "neutral": 0.0},
    "knockout": {"is_knockout": 1.0, "neutral": 0.0},
    "final": {"is_knockout": 1.0, "neutral": 1.0},
}
MAPPING_PATH = dc.DATA_DIR / "sim_team_mapping.csv"
EXPORT_DIR = dc.PROJECT_ROOT / "exports"
MAPPING_COLUMNS = ["competition", "sim_name", "sim_country", "key", "model_name", "model_country", "method"]


# =============================================================================
# Takım eşleştirmesi
# =============================================================================

def load_sim_teams(teams_dir: Path) -> pd.DataFrame:
    rows = []
    for comp in COMPETITIONS:
        path = teams_dir / comp / f"{comp}_teams.json"
        if not path.exists():
            raise FileNotFoundError(f"Takım listesi bulunamadı: {path}")
        pots = json.loads(path.read_text(encoding="utf-8"))
        for pot in pots.values():
            for team in pot:
                rows.append({"competition": comp, "sim_name": team["name"], "sim_country": team["country"]})
    return pd.DataFrame(rows)


def _country_ok(model_country, sim_country) -> bool:
    """Modelde ülkesi bilinmeyen kulüp (iç ligi veride olmayan ülkeler) her ülke kodu için kabul edilir."""
    return model_country is None or (isinstance(model_country, float) and np.isnan(model_country)) \
        or model_country == sim_country


def auto_resolve(name: str, country: str, identity: ft.TeamIdentity) -> tuple[str | None, str, list[str]]:
    """(anahtar, yöntem, öneriler). Ülkesi tutmayan eşleşme ASLA kabul edilmez."""
    key, info = identity.resolve(name)
    if key is not None and _country_ok(identity.country.get(key), country):
        return key, f"auto_{info}", []
    # Aynı ülkenin (ya da ülkesi bilinmeyen) kulüpleri arasında ad benzerliği
    pool = {k: v for k, v in identity.display.items() if _country_ok(identity.country.get(k), country)
            and k.startswith("tm:")}
    key, method = ft._unique_name_match(name, pool)
    if key is not None:
        return key, f"auto_country_{method}", []
    import difflib
    same_country = {k: v for k, v in pool.items() if identity.country.get(k) == country} or pool
    ranked = sorted(same_country.items(),
                    key=lambda kv: -difflib.SequenceMatcher(None, name.lower(), kv[1].lower()).ratio())[:5]
    return None, "unresolved", [f"{v} ({k})" for k, v in ranked]


def build_mapping(teams: pd.DataFrame, identity: ft.TeamIdentity) -> tuple[pd.DataFrame, dict]:
    existing = pd.read_csv(MAPPING_PATH, dtype=str) if MAPPING_PATH.exists() else pd.DataFrame(columns=MAPPING_COLUMNS)
    existing = existing.dropna(subset=["key"])
    # Dosyadaki satırlar (elle girilmiş ya da önceki çalıştırmada otomatik bulunup gözden geçirilmiş) korunur
    manual = {(r.competition, r.sim_name): (r.key, r.method if isinstance(r.method, str) else "manual")
              for r in existing.itertuples()}
    rows, unresolved = [], {}
    for t in teams.itertuples():
        if (t.competition, t.sim_name) in manual:
            key, method = manual[(t.competition, t.sim_name)]
            if key not in identity.display:
                raise ValueError(f"sim_team_mapping.csv: '{t.sim_name}' için anahtar modelde yok: {key}")
            suggestions = []
        else:
            key, method, suggestions = auto_resolve(t.sim_name, t.sim_country, identity)
        if key is None:
            unresolved[f"{t.competition}:{t.sim_name} ({t.sim_country})"] = suggestions
        rows.append({"competition": t.competition, "sim_name": t.sim_name, "sim_country": t.sim_country,
                     "key": key, "model_name": identity.display.get(key) if key else None,
                     "model_country": identity.country.get(key) if key else None, "method": method})
    mapping = pd.DataFrame(rows, columns=MAPPING_COLUMNS)
    # Aynı turnuvada iki farklı simülasyon takımı aynı kulübe düşerse bu bir hatadır
    dup = mapping.dropna(subset=["key"]).duplicated(["competition", "key"], keep=False)
    if dup.any():
        raise ValueError("Aynı turnuvada iki takım aynı kulübe eşlendi:\n"
                         + mapping.dropna(subset=["key"])[dup].to_string(index=False))
    return mapping, unresolved


# =============================================================================
# Dışa aktarım
# =============================================================================

def build_fixtures(mapping: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for comp, grp in mapping.groupby("competition", sort=False):
        teams = list(zip(grp["sim_name"], grp["key"]))
        for (h_name, h_key) in teams:
            for (a_name, a_key) in teams:
                if h_name == a_name:
                    continue
                for stage, flags in STAGES.items():
                    rows.append({"competition": comp, "stage": stage, "home": h_name, "away": a_name,
                                 "date": when, "home_key": h_key, "away_key": a_key, "is_uefa": 1.0, **flags})
    return pd.DataFrame(rows)


class ExportError(RuntimeError):
    """Tablo güvenle üretilemedi (eşlenemeyen takım, Elo'su olmayan takım); tablo YAZILMAZ."""


LATEST_EXPORT_PATH = EXPORT_DIR / "sim_predictions_latest.json"


def run_export(teams_dir: Path, when: pd.Timestamp, predictor: MatchPredictor | None = None,
               out: Path | None = None, full_matrix: bool = False, write_latest: bool = True) -> dict:
    """
    Tahmin tablosunu üretir ve yazar. Haftalık güncelleme bu fonksiyonu çağırır.
    Dönüş: özet (dosya yolu, tahmin sayısı, eşleştirme yöntemleri, NaN oranları).
    """
    predictor = predictor or MatchPredictor()
    teams = load_sim_teams(Path(teams_dir))
    mapping, unresolved = build_mapping(teams, predictor.store.identity)
    mapping.to_csv(MAPPING_PATH, index=False)
    log.info("Takım eşleştirmesi yazıldı: %s (%s)", MAPPING_PATH, mapping["method"].value_counts().to_dict())
    if unresolved:
        lines = "\n".join(f"  {name}: adaylar {cands}" for name, cands in unresolved.items())
        raise ExportError(f"{len(unresolved)} takım eşlenemedi — {MAPPING_PATH.name} dosyasına 'key' girin "
                          f"(ör. tm:131):\n{lines}")

    fixtures = build_fixtures(mapping, when)
    log.info("%d tahmin hesaplanıyor (referans tarih %s)...", len(fixtures), when.date())
    pred = predictor.predict_many(fixtures)
    # is_knockout / is_uefa hem fikstürde hem özelliklerde var; tek kopya kalsın
    res = pd.concat([fixtures.reset_index(drop=True),
                     pred.drop(columns=[c for c in pred.columns if c in fixtures.columns])], axis=1)
    # Elo'su olmayan takım (son ELO_MAX_STALENESS_DAYS günde veri setinde maçı yok): o takımın eşleşmeleri
    # tabloya YAZILMAZ ve meta'da listelenir; simülasyon bu takımlar için kendi formülüne döner.
    # Diğer tüm tahminler yine güncellenir (tek takım yüzünden tablo eskimez).
    no_elo: list[str] = []
    if res["elo_missing"].any():
        for comp, grp in mapping.groupby("competition"):
            probe = predictor.store.compute(pd.DataFrame({
                "date": when, "home_key": grp["key"].to_numpy(), "away_key": grp["key"].to_numpy(), "is_uefa": 1.0}))
            no_elo += [f"{comp}:{n}" for n, e in zip(grp["sim_name"], probe["home_elo"]) if np.isnan(e)]
        log.warning("Elo'su olmayan takımlar (son %d günde maçı yok), eşleşmeleri tabloya yazılmıyor: %s",
                    ft.ELO_MAX_STALENESS_DAYS, no_elo)
        res = res[~res["elo_missing"]].reset_index(drop=True)
    if res.empty:
        raise ExportError("Hiçbir eşleşme için tahmin üretilemedi (Elo yok).")

    ens = predictor.bundle["ensemble"]
    nan_rates = {c: round(float(res[c].isna().mean()), 4) for c in predictor.feature_columns}
    predictions: dict = {}
    for r in res.itertuples():
        entry = {"p": [round(r.home_win, 6), round(r.draw, 6), round(r.away_win, 6)],
                 "lambda": [round(r.lambda_home, 6), round(r.lambda_away, 6)]}
        if full_matrix:
            entry["score_matrix"] = predictor.predict_score(r.home_key, r.away_key, when,
                                                            neutral=bool(r.neutral), is_knockout=bool(r.is_knockout),
                                                            is_uefa=True)["score_matrix"]
        predictions.setdefault(r.competition, {}).setdefault(r.home, {}).setdefault(r.away, {})[r.stage] = entry

    schema = predictor.schema
    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "reference_date": str(when.date()),
            "model_trained_at_utc": schema["trained_at"],
            "model_fitted_on": schema["fitted_on"],
            "model_data_end": schema.get("data_end"),
            "contract_hash": schema["contract_hash"],
            "data_last_match_by_source": {k: str(v.date()) for k, v in predictor.store.last_match_by_source.items()},
            "stages": STAGES,
            "entry_format": {
                "p": "[ev sahibi kazanır, beraberlik, deplasman kazanır] olasılıkları (topluluk modelinin nihai çıktısı)",
                "lambda": "[λ_ev, λ_deplasman] Poisson modellerinin beklenen golleri (90 dakika)",
                "score_matrix": "(yalnızca --full-matrix) [i][j] = ev i - deplasman j gol olasılığı",
            },
            "score_matrix_recipe": [
                f"i, j = 0..{MAX_GOALS}; M[i][j] = Poisson(i; λ_ev) * Poisson(j; λ_dep)",
                "M[0][0] *= 1 - λ_ev*λ_dep*rho; M[0][1] *= 1 + λ_ev*rho; M[1][0] *= 1 + λ_dep*rho; M[1][1] *= 1 - rho",
                "negatifleri 0 yap, M'yi toplamı 1 olacak şekilde normalize et",
                "i>j hücrelerini p[0]/Σ(i>j), i==j hücrelerini p[1]/Σ(i==j), i<j hücrelerini p[2]/Σ(i<j) ile çarp",
            ],
            "rho": ens["rho"],
            "max_goals": MAX_GOALS,
            "extra_time_hint": "Uzatma (30 dk) için yaklaşık λ * 30/90; model uzatma verisiyle eğitilmedi.",
            "feature_nan_rates": nan_rates,
            "teams_without_prediction": no_elo,
            "teams_without_prediction_reason": (
                f"Son {ft.ELO_MAX_STALENESS_DAYS} günde veri setinde maçı yok (Elo hesaplanamaz). Bu takımların "
                "eşleşmeleri tabloda yoktur; simülasyon kendi formülünü kullanmalıdır."),
        },
        "teams": {comp: grp.drop(columns=["competition"]).where(grp.notna(), None).to_dict("records")
                  for comp, grp in mapping.groupby("competition", sort=False)},
        "predictions": predictions,
    }
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out) if out else EXPORT_DIR / f"sim_predictions_{when.date()}.json"
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out.write_text(text, encoding="utf-8")
    if write_latest:
        # Önce geçici dosyaya yaz, sonra yer değiştir: simülasyon yarım yazılmış dosya okumasın
        tmp = LATEST_EXPORT_PATH.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(LATEST_EXPORT_PATH)
    log.info("Yazıldı: %s (%.1f MB, %d tahmin)", out, out.stat().st_size / 1e6, len(res))
    return {"path": str(out), "n_predictions": int(len(res)), "reference_date": str(when.date()),
            "mapping_methods": mapping["method"].value_counts().to_dict(), "feature_nan_rates": nan_rates,
            "teams_without_prediction": no_elo}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simülasyon için tahmin tablosu")
    parser.add_argument("--teams-dir", type=Path, required=True,
                        help="Simülasyonun data klasörü ({turnuva}/{turnuva}_teams.json içerir)")
    parser.add_argument("--date", type=pd.Timestamp, default=pd.Timestamp(date.today()),
                        help="Özelliklerin hesaplanacağı referans tarih (bu tarihten önceki veri kullanılır)")
    parser.add_argument("--full-matrix", action="store_true", help="11x11 skor matrislerini de yaz")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_export(args.teams_dir, args.date, out=args.out, full_matrix=args.full_matrix)
    except ExportError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
