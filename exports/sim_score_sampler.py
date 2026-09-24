"""
Simülasyon tarafı için referans uygulama: sim_predictions_<tarih>.json tablosundan skor olasılık matrisini
kurar ve skor örnekler. Bilerek yalnızca Python standart kütüphanesi kullanır (numpy / scipy gerekmez),
böylece simülasyon projesine yeni bağımlılık eklemeden kopyalanabilir.

    table = load_table("sim_predictions_2026-09-14.json")
    entry = get_entry(table, "ucl", "Galatasaray", "Bayern München", "league")
    home_goals, away_goals = sample_score(entry, table["meta"]["rho"])

Matris formülü futbol-ml-modeli/src/predict.py içindeki dixon_coles_matrix + consistent_score_matrix ile
birebir aynıdır (tests: futbol-ml-modeli README'deki doğrulama). Değiştirmeyin.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

MAX_GOALS = 10


def load_table(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_entry(table: dict, competition: str, home: str, away: str, stage: str) -> dict:
    """
    stage: "league" (lig aşaması), "knockout" (eleme turu maçı), "final" (tek maç, tarafsız saha).
    Takım adları simülasyonun kendi adlarıdır (ör. "Bayern München").
    """
    try:
        return table["predictions"][competition][home][away][stage]
    except KeyError as exc:
        raise KeyError(f"Tabloda yok: {competition} / {home} - {away} / {stage}") from exc


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else float(k == 0)


def score_matrix(entry: dict, rho: float, max_goals: int = MAX_GOALS) -> list[list[float]]:
    """[i][j] = ev sahibi i, deplasman j gol olasılığı; G/B/M bölge toplamları entry['p'] ile aynıdır."""
    lam_h, lam_a = entry["lambda"]
    ph = [_poisson_pmf(i, lam_h) for i in range(max_goals + 1)]
    pa = [_poisson_pmf(j, lam_a) for j in range(max_goals + 1)]
    m = [[ph[i] * pa[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    # Dixon-Coles düşük skor düzeltmesi
    m[0][0] *= 1 - lam_h * lam_a * rho
    m[0][1] *= 1 + lam_h * rho
    m[1][0] *= 1 + lam_a * rho
    m[1][1] *= 1 - rho
    m = [[max(0.0, v) for v in row] for row in m]
    total = sum(map(sum, m))
    m = [[v / total for v in row] for row in m]
    # Ev kazanır / beraberlik / deplasman kazanır bölgelerini modelin nihai olasılıklarına ölçekle
    region_sum = [0.0, 0.0, 0.0]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            region_sum[0 if i > j else 1 if i == j else 2] += m[i][j]
    scale = [p / max(s, 1e-6) for p, s in zip(entry["p"], region_sum)]
    return [[m[i][j] * scale[0 if i > j else 1 if i == j else 2] for j in range(max_goals + 1)]
            for i in range(max_goals + 1)]


def sample_score(entry: dict, rho: float, rng: random.Random | None = None) -> tuple[int, int]:
    """Skor matrisinden bir (ev, deplasman) skoru örnekler."""
    rng = rng or random
    m = score_matrix(entry, rho)
    u, acc = rng.random(), 0.0
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            acc += v
            if u < acc:
                return i, j
    return MAX_GOALS, MAX_GOALS  # yuvarlama payı


def extra_time_entry(entry: dict) -> dict:
    """
    Uzatma (30 dk) için yaklaşık: beklenen goller 30/90 ile ölçeklenir, skor bağımsız Poisson olarak çekilir
    (G/B/M ölçeklemesi uygulanmaz). Model uzatma verisiyle eğitilmedi; bu bir yaklaşıklıktır.
    """
    lam_h, lam_a = entry["lambda"]
    return {"lambda": [lam_h * 30 / 90, lam_a * 30 / 90]}


def sample_extra_time(entry: dict, rng: random.Random | None = None) -> tuple[int, int]:
    rng = rng or random
    lam_h, lam_a = extra_time_entry(entry)["lambda"]

    def draw(lam):
        # Knuth yöntemi (küçük λ için yeterli)
        limit, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1

    return draw(lam_h), draw(lam_a)
