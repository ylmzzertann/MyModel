"""
Haftalık otomatik güncelleme. Windows Görev Zamanlayıcı her hafta run_weekly.bat ile çalıştırır.

    python -m src.weekly_update                  # standart haftalık akış
    python -m src.weekly_update --force-retrain  # yeniden eğitimi şimdi zorla
    python -m src.weekly_update --no-retrain     # bu çalıştırmada yeniden eğitimi atla

Neden her hafta yeniden eğitim YOK: Model, takımların güncel durumunu özelliklerden (Elo, piyasa Elo'su,
form, şut payı, ilk 11 değeri) okur. Bu özellikler her hafta yeni maçlarla yeniden hesaplanır; yani
tahminler her hafta güncellenir. Modelin "özellik -> olasılık" eşlemesi ise yavaş değişir; onu yeniden
öğrenmek (~50 dk) ayda bir, yeni sezonda ya da canlı performans bozulduğunda yeterlidir.

Adımlar
-------
1. Kilit: aynı anda iki güncelleme çalışmaz.
2. Veri: football-data (devam eden sezon + extra ligler) ve Transfermarkt (sunucuda yeni sürüm varsa).
3. Özellik deposu yeni veriyle yeniden kurulur (üretim modelinin Elo parametreleriyle; model değişmez).
4. Canlı performans: modelin eğitim verisinin bittiği tarihten SONRA oynanan 1. lig + UEFA maçlarında,
   maç öncesi özelliklerle yapılan tahminlerin log-loss / Brier / accuracy'si ve kapanış oranlarıyla kıyas.
5. Yeniden eğitim kararı (biri yeterli):
     - model RETRAIN_EVERY_DAYS günden eski,
     - yeni sezon başladı (modelin test sezonu artık "son tamamlanmış sezon" değil),
     - canlı performans bozuldu (>= MIN_LIVE_MATCHES maçta piyasaya göre fark, testteki farktan
       LIVE_GAP_TOLERANCE'tan fazla).
   Eğitim models/staging/ klasöründe yapılır (--refit-all). Kalite kapısı geçilirse mevcut model
   models/arsiv/<zaman>/ klasörüne kopyalanır ve yeni model üretime alınır; geçilmezse mevcut model kalır.
6. Simülasyon tablosu: exports/sim_predictions_<bugün>.json + exports/sim_predictions_latest.json
7. Rapor: reports/haftalik_rapor.md (son çalıştırma), reports/guncelleme_gecmisi.csv,
   models/canli_performans.csv, logs/guncelleme_<tarih>.log

Çıkış kodu: 0 = sorunsuz, 2 = tamamlandı ama uyarı var, 1 = kritik hata (tablo güncellenemedi).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_collection as dc
from . import features as ft

log = logging.getLogger("weekly_update")

ROOT = dc.PROJECT_ROOT
MODELS_DIR = ROOT / "models"
STAGING_DIR = MODELS_DIR / "staging"
ARCHIVE_DIR = MODELS_DIR / "arsiv"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
LOCK_PATH = ROOT / ".guncelleme.lock"
LIVE_PERF_PATH = MODELS_DIR / "canli_performans.csv"
HISTORY_PATH = REPORTS_DIR / "guncelleme_gecmisi.csv"
REPORT_PATH = REPORTS_DIR / "haftalik_rapor.md"

# Simülasyonun takım listelerinin bulunduğu klasör (SIM_TEAMS_DIR ortam değişkeniyle değiştirilebilir)
DEFAULT_SIM_TEAMS_DIR = Path(os.environ.get(
    "SIM_TEAMS_DIR", r"C:\Users\ertan\OneDrive\Desktop\Champions League Simulator\API\data"))

RETRAIN_EVERY_DAYS = 28
MIN_LIVE_MATCHES = 300
LIVE_GAP_TOLERANCE = 0.02      # canlı (model - piyasa) log-loss farkı, testteki farkı bu kadar aşarsa
QUALITY_TOLERANCE = 0.01       # yeni modelin CV ölçütü, mevcudunkinden en fazla bu kadar kötü olabilir
LOCK_MAX_AGE_HOURS = 6
PRODUCTION_FILES = ["mac_modeli.pkl", "metrics.json", "calibration_report.csv", "elo_tuning.csv",
                    "feature_selection.csv", "hyperparameter_search.csv"]


# =============================================================================
# Yardımcılar
# =============================================================================

def acquire_lock() -> None:
    if LOCK_PATH.exists():
        age_h = (time.time() - LOCK_PATH.stat().st_mtime) / 3600
        if age_h < LOCK_MAX_AGE_HOURS:
            raise RuntimeError(f"Başka bir güncelleme çalışıyor olabilir ({LOCK_PATH}, {age_h:.1f} saat önce). "
                               f"Takılı kaldıysa dosyayı silin.")
        log.warning("Eski kilit dosyası (%.1f saat) yok sayılıyor.", age_h)
    LOCK_PATH.write_text(f"{os.getpid()} {datetime.now().isoformat()}", encoding="utf-8")


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def production_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def model_data_end(schema: dict) -> pd.Timestamp:
    """Modelin eğitildiği son maç tarihi (eski şemalarda data_range'den çıkarılır)."""
    if schema.get("data_end"):
        return pd.Timestamp(schema["data_end"])
    rng = schema["data_range"]
    part = "test" if "refit-all" in schema.get("fitted_on", "") else "development"
    return pd.Timestamp(rng[part]["to"])


# =============================================================================
# Adımlar
# =============================================================================

def update_data(report: dict) -> None:
    for name, fn in (("football-data", dc.download_football_data), ("transfermarkt", dc.download_transfermarkt)):
        try:
            files = fn()
            report["data"][name] = f"{len(files)} dosya hazır"
        except Exception as exc:  # noqa: BLE001 - bir kaynak inmezse eldeki veriyle devam edilir
            report["data"][name] = f"İNDİRİLEMEDİ: {exc}"
            report["warnings"].append(f"{name} indirilemedi; mevcut veriyle devam edildi.")
            log.exception("%s indirilemedi", name)


def live_performance(predictor, since: pd.Timestamp) -> dict:
    """Modelin eğitim verisinden SONRAKİ maçlarda, maç öncesi özelliklerle tahmin başarısı."""
    from .train import evaluate, market_reference

    fd, tm = dc.load_domestic_matches(), dc.load_transfermarkt_games()
    m = ft.drop_same_day_duplicates(ft.assemble_matches(fd, tm, predictor.store.identity))
    m = m[(m["date"] > since) & ((m["tier"] == 1) | m["is_uefa"])].reset_index(drop=True)
    if m.empty:
        return {"n": 0, "since": str(since.date())}
    fx = m[["date", "home_key", "away_key"]].assign(
        neutral=m["neutral"].astype(float), is_knockout=m["is_knockout"].astype(float),
        is_uefa=m["is_uefa"].astype(float))
    pred = predictor.predict_many(fx)
    ok = ~pred["elo_missing"].to_numpy()
    y = m.loc[ok, "result"].map(ft.RESULT_TO_CLASS).to_numpy()
    p = pred.loc[ok, ["home_win", "draw", "away_win"]].to_numpy()
    out = {"since": str(since.date()), "until": str(m["date"].max().date()), **evaluate(y, p)}
    uefa = m.loc[ok, "is_uefa"].to_numpy(bool)
    if uefa.sum() >= 30:
        out["uefa"] = evaluate(y[uefa], p[uefa])
    market = market_reference(m[ok].reset_index(drop=True))
    if market is not None:
        mask, p_market = market
        out["market_subset_n"] = int(mask.sum())
        out["model_log_loss_on_market_subset"] = evaluate(y[mask], p[mask])["log_loss"]
        out["market_log_loss"] = evaluate(y[mask], p_market)["log_loss"]
        out["gap_to_market"] = out["model_log_loss_on_market_subset"] - out["market_log_loss"]
    return out


def retrain_reasons(schema: dict, metrics: dict, live: dict, today: date) -> list[str]:
    from .train import default_test_from

    reasons = []
    trained = pd.Timestamp(schema["trained_at"]).date()
    age = (today - trained).days
    if age >= RETRAIN_EVERY_DAYS:
        reasons.append(f"model {age} gün önce eğitildi (sınır {RETRAIN_EVERY_DAYS})")
    test_from = metrics.get("data_range", {}).get("test", {}).get("from")
    if test_from and pd.Timestamp(test_from) != default_test_from(today):
        reasons.append(f"yeni sezon: test sezonu {test_from} -> {default_test_from(today).date()}")
    test_gap = None
    tm = metrics.get("test_metrics", {}).get("ic_lig", {})
    if "ensemble_on_market_subset" in tm and "market_odds_reference" in tm:
        test_gap = tm["ensemble_on_market_subset"]["log_loss"] - tm["market_odds_reference"]["log_loss"]
    if (test_gap is not None and live.get("market_subset_n", 0) >= MIN_LIVE_MATCHES
            and live["gap_to_market"] > test_gap + LIVE_GAP_TOLERANCE):
        reasons.append(f"canlı performans bozuldu: piyasa farkı {live['gap_to_market']:.4f} "
                       f"(testte {test_gap:.4f}, tolerans {LIVE_GAP_TOLERANCE})")
    return reasons


def retrain_and_promote(report: dict) -> bool:
    """Staging'de eğitir, kalite kapısından geçerse üretime alır. Dönüş: yeni model devreye alındı mı."""
    from .predict import MatchPredictor

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    STAGING_DIR.mkdir(parents=True)
    train_log = LOGS_DIR / f"egitim_{datetime.now():%Y-%m-%d_%H%M}.log"
    log.info("Yeniden eğitim başlıyor (staging: %s, log: %s)...", STAGING_DIR, train_log)
    with open(train_log, "w", encoding="utf-8") as fh:
        proc = subprocess.run([sys.executable, "-m", "src.train", "--refit-all", "--output-dir", str(STAGING_DIR)],
                              cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        report["retrain"]["result"] = f"EĞİTİM BAŞARISIZ (kod {proc.returncode}); log: {train_log.name}"
        report["warnings"].append("Yeniden eğitim başarısız; mevcut model kullanılmaya devam ediyor.")
        return False

    # Kalite kapısı
    new_metrics = json.loads((STAGING_DIR / "metrics.json").read_text(encoding="utf-8"))
    old_metrics = production_metrics()
    checks = []
    try:
        cand = MatchPredictor(model_path=STAGING_DIR / "mac_modeli.pkl", schema_path=STAGING_DIR / "feature_schema.md")
        probe = cand.predict("tm:27", "tm:131", pd.Timestamp(date.today()), is_uefa=True)
        checks.append(("yeni model yüklenip tahmin yapabiliyor",
                       np.isclose(sum(probe.values()), 1.0) and all(0 < v < 1 for v in probe.values())))
    except Exception as exc:  # noqa: BLE001
        checks.append((f"yeni model yüklenip tahmin yapabiliyor ({exc})", False))
    new_cv = new_metrics["cv_objective"]["objective"]
    old_cv = old_metrics.get("cv_objective", {}).get("objective")
    checks.append((f"CV ölçütü {new_cv:.5f} <= mevcut {old_cv if old_cv is None else round(old_cv, 5)} + {QUALITY_TOLERANCE}",
                   old_cv is None or new_cv <= old_cv + QUALITY_TOLERANCE))
    t = new_metrics["test_metrics"]["tum_maclar"]
    checks.append((f"test log-loss {t['ensemble']['log_loss']:.4f} < naif {t['naive_class_frequencies']['log_loss']:.4f} - 0.03",
                   t["ensemble"]["log_loss"] < t["naive_class_frequencies"]["log_loss"] - 0.03))
    report["retrain"]["checks"] = [(name, bool(ok)) for name, ok in checks]
    if not all(ok for _, ok in checks):
        report["retrain"]["result"] = "KALİTE KAPISI GEÇİLEMEDİ; mevcut model korundu (yeni model models/staging/)"
        report["warnings"].append("Yeni model kalite kapısını geçemedi.")
        return False

    # Arşivle ve üretime al
    archive = ARCHIVE_DIR / datetime.now().strftime("%Y-%m-%d_%H%M")
    archive.mkdir(parents=True)
    for name in PRODUCTION_FILES:
        if (MODELS_DIR / name).exists():
            shutil.copy2(MODELS_DIR / name, archive / name)
    shutil.copy2(ROOT / "feature_schema.md", archive / "feature_schema.md")
    for name in PRODUCTION_FILES:
        if (STAGING_DIR / name).exists():
            shutil.copy2(STAGING_DIR / name, MODELS_DIR / name)
    shutil.copy2(STAGING_DIR / "feature_schema.md", ROOT / "feature_schema.md")
    report["retrain"]["result"] = f"YENİ MODEL DEVREDE (önceki model arşivde: models/arsiv/{archive.name})"
    report["retrain"]["new_cv_objective"] = new_cv
    return True


# =============================================================================
# Rapor
# =============================================================================

def write_report(report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    live = report.get("live", {})
    lines = [
        f"# Haftalık güncelleme raporu — {report['started']}",
        "",
        f"**Durum:** {report['status']}",
        "",
        "## Veri",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in report["data"].items()]
    lines += [f"- Kaynakların son maç tarihleri: {report.get('last_match_by_source', '-')}", "",
              "## Model", "", f"- Eğitim zamanı: {report.get('model_trained_at', '-')}",
              f"- Eğitim verisinin son maçı: {report.get('model_data_end', '-')}", "",
              "## Canlı performans (eğitimden sonra oynanan maçlar)", ""]
    if live.get("n"):
        lines += [f"- Dönem: {live['since']} sonrası → {live['until']}, {live['n']} maç",
                  f"- Log-loss {live['log_loss']:.4f}, Brier {live['brier']:.4f}, accuracy {live['accuracy']:.3f}"]
        if "uefa" in live:
            u = live["uefa"]
            lines.append(f"- UEFA ({u['n']} maç): log-loss {u['log_loss']:.4f}, accuracy {u['accuracy']:.3f}")
        if "gap_to_market" in live:
            lines.append(f"- Kapanış oranlarıyla kıyas ({live['market_subset_n']} maç): model "
                         f"{live['model_log_loss_on_market_subset']:.4f}, piyasa {live['market_log_loss']:.4f}, "
                         f"fark {live['gap_to_market']:+.4f}")
        if live["n"] < MIN_LIVE_MATCHES:
            lines.append(f"- Not: {MIN_LIVE_MATCHES} maçtan az; değerler henüz gürültülü.")
    else:
        lines.append("- Henüz eğitimden sonra oynanmış maç yok.")
    rt = report["retrain"]
    lines += ["", "## Yeniden eğitim", "",
              f"- Gerekçe: {', '.join(rt['reasons']) if rt['reasons'] else 'gerek yok'}",
              f"- Sonuç: {rt.get('result', 'yapılmadı')}"]
    lines += [f"  - {'✔' if ok else '✘'} {name}" for name, ok in rt.get("checks", [])]
    lines += ["", "## Simülasyon tablosu", "", f"- {report.get('export', '-')}", ""]
    if report["warnings"]:
        lines += ["## Uyarılar", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    if report.get("error"):
        lines += ["## Hata", "", "```text", report["error"], "```", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    row = {"started": report["started"], "status": report["status"],
           "live_n": live.get("n", 0), "live_log_loss": live.get("log_loss"),
           "live_gap_to_market": live.get("gap_to_market"),
           "retrain": rt.get("result", "yapılmadı"), "warnings": " | ".join(report["warnings"])}
    pd.DataFrame([row]).to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)
    if live.get("n"):
        live_row = {"run": report["started"], **{k: v for k, v in live.items() if not isinstance(v, dict)}}
        pd.DataFrame([live_row]).to_csv(LIVE_PERF_PATH, mode="a", header=not LIVE_PERF_PATH.exists(), index=False)


# =============================================================================
# Ana akış
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Haftalık otomatik güncelleme")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Veri indirmeyi atla (test için)")
    parser.add_argument("--sim-teams-dir", type=Path, default=DEFAULT_SIM_TEAMS_DIR)
    args = parser.parse_args(argv)

    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(LOGS_DIR / f"guncelleme_{date.today()}.log", encoding="utf-8")])
    report = {"started": datetime.now().strftime("%Y-%m-%d %H:%M"), "status": "başladı", "data": {},
              "warnings": [], "retrain": {"reasons": []}}
    today = date.today()
    exit_code = 0
    try:
        acquire_lock()
    except RuntimeError as exc:
        log.error(str(exc))
        return 1
    try:
        from .export_predictions import ExportError, run_export
        from .predict import MatchPredictor

        if args.skip_download:
            report["data"]["indirme"] = "atlandı (--skip-download)"
        else:
            log.info("1/5 Veri indiriliyor...")
            update_data(report)

        log.info("2/5 Özellik deposu yeni veriyle kuruluyor...")
        predictor = MatchPredictor()
        report["last_match_by_source"] = {k: str(v.date()) for k, v in predictor.store.last_match_by_source.items()}
        report["model_trained_at"] = predictor.schema["trained_at"]
        data_end = model_data_end(predictor.schema)
        report["model_data_end"] = str(data_end.date())

        log.info("3/5 Canlı performans ölçülüyor (%s sonrası maçlar)...", data_end.date())
        report["live"] = live_performance(predictor, data_end)

        log.info("4/5 Yeniden eğitim kararı...")
        reasons = retrain_reasons(predictor.schema, production_metrics(), report["live"], today)
        if args.force_retrain:
            reasons.append("elle istendi (--force-retrain)")
        report["retrain"]["reasons"] = reasons
        if reasons and not args.no_retrain:
            if retrain_and_promote(report):
                predictor = MatchPredictor()   # yeni üretim modeli
                report["model_trained_at"] = predictor.schema["trained_at"]
                report["model_data_end"] = str(model_data_end(predictor.schema).date())
        elif reasons:
            report["retrain"]["result"] = "gerekçe var ama --no-retrain ile atlandı"

        log.info("5/5 Simülasyon tahmin tablosu üretiliyor...")
        if args.sim_teams_dir.exists():
            summary = run_export(args.sim_teams_dir, pd.Timestamp(today), predictor=predictor)
            report["export"] = (f"{Path(summary['path']).name} + sim_predictions_latest.json "
                                f"({summary['n_predictions']} tahmin, referans tarih {summary['reference_date']})")
            if summary["teams_without_prediction"]:
                report["warnings"].append(
                    f"Elo'su olmayan (son {ft.ELO_MAX_STALENESS_DAYS} günde maçı olmayan) takımlar tabloya yazılmadı, "
                    f"simülasyon bunlar için kendi formülünü kullanır: {summary['teams_without_prediction']}")
        else:
            report["export"] = f"ATLANDI: simülasyon klasörü bulunamadı ({args.sim_teams_dir})"
            report["warnings"].append("Simülasyon takım listesi bulunamadı; tablo güncellenmedi.")
        report["status"] = "TAMAMLANDI" + (" (uyarılarla)" if report["warnings"] else "")
        exit_code = 2 if report["warnings"] else 0
    except Exception as exc:  # noqa: BLE001 - her hata rapora yazılır
        if exc.__class__.__name__ == "ExportError":
            report["warnings"].append("Tahmin tablosu üretilemedi (takım eşleştirmesi / Elo); ayrıntı aşağıda.")
        report["status"] = "HATA"
        report["error"] = traceback.format_exc()
        log.exception("Güncelleme başarısız")
        exit_code = 1
    finally:
        write_report(report)
        release_lock()
        log.info("Rapor: %s — durum: %s", REPORT_PATH, report["status"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
