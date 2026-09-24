"""
Özellik mühendisliği.

EN ÖNEMLİ KURAL: Eğitim (train.py) ve tahmin (predict.py) özellikleri AYNI
fonksiyonlarla hesaplar (`FeatureStore.compute`). Böylece eğitim/kullanım
arasında sessiz hesaplama farkı oluşmaz. Her özelliğin tanımı `FEATURE_SPECS`
içinde durur ve feature_schema.md bu listeden otomatik üretilir.

Bileşenler
----------
1. Takım kimliği: kanonik anahtar Transfermarkt kulüp id'sidir ("tm:131").
   football-data adları bu id'lere, AYNI GÜN AYNI SKORLA oynanmış maçların
   çakıştırılmasıyla eşlenir (bulanık ad eşleştirmesinden çok daha güvenilir).
   Eşlenemeyenler "fd:ÜLKE:normalize_ad" anahtarıyla ayrı kimlik olarak kalır.
2. Ligler arası Elo: iç lig + UEFA maçları tarih sırasıyla işlenir. Ligler arası
   güç farkı UEFA maçları üzerinden akar (ClubElo'nun yaklaşımı).
3. Form: son 5 maçın puan ortalaması, son 10 maçın attığı / yediği gol ortalaması.
4. Kadro değeri: Transfermarkt piyasa değerleri (en değerli 16 oyuncu).
5. Oyuncu bazlı takım gücü: kulübün son 5 maçında en çok süre alan "düzenli ilk 11"in
   piyasa değeri, yıldız oyuncusu, yedek derinliği, hat (hücum/orta saha/defans/kaleci)
   değerleri, gol+asist üretimi, yaş, UEFA tecrübesi, son maçtaki rotasyon; maçlar arası dinlenme.

Özellikler GRUPLAR halinde tanımlıdır; hangi grupların modele gireceğine train.py doğrulama
setinde ileri seçimle (forward selection) karar verir.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data_collection as dc

log = logging.getLogger(__name__)

# =============================================================================
# Sabitler — feature_schema.md'ye aynen yazılır; predict.py şemadaki değerlerle
# buradaki değerlerin birebir aynı olduğunu doğrular.
# =============================================================================

ELO_BASE = 1500.0                  # ilk rating (veri başında ve havuz yetersizken)
HOME_ELO_ADVANTAGE = 55.0          # ev sahibine eklenen Elo puanı (tarafsız sahada 0)
ELO_MAX_STALENESS_DAYS = 400       # kulübün son maçı sorgudan bu kadar eskiyse -> NaN
ELO_PROVISIONAL_MATCHES = 10       # yeni kulübün ilk N maçında K çarpanı uygulanır
ELO_PROVISIONAL_K_MULT = 2.0
ELO_NEWCOMER_PERCENTILE = 25       # yeni kulüp = ülkesindeki mevcut ratinglerin 25. yüzdeliği
ELO_NEWCOMER_MIN_POOL = 6          # havuzda bundan az kulüp varsa ELO_BASE
ELO_BURN_IN_PASSES = 3             # ısınma döneminin (TRAIN_FIRST_SEASON öncesi) tekrar sayısı
# Eğitimde ayarlanan Elo parametrelerinin arama ızgarası (sonuç şemaya yazılır)
ELO_K_GRID = (5.0, 7.5, 10.0, 15.0, 20.0)
ELO_UEFA_K_MULT_GRID = (1.0, 2.0, 3.0, 4.0, 6.0)
# Piyasa Elo'su: oranı olan maçlarda hedef, oranların ima ettiği beklenen skorla harmanlanır
MARKET_K_GRID = (5.0, 10.0, 15.0, 20.0, 30.0)
MARKET_WEIGHT_GRID = (0.5, 0.75, 1.0)

SHOTS_WINDOW = 10                  # isabetli şut / şut payı için şut verili son maç sayısı
SHOTS_MIN_MATCHES = 5

FORM_WINDOW = 5                    # son kaç maçın puan ortalaması
FORM_MIN_MATCHES = 3               # pencerede en az bu kadar maç yoksa -> NaN
FORM_MAX_GAP_DAYS = 200            # ardışık iki maç arası bundan uzunsa form "sıfırlanır"

SQUAD_WINDOW_DAYS = 180            # ay başından önceki 180 günde kulüp için forma giyen oyuncular
SQUAD_TOP_N = 16                   # en değerli 16 oyuncunun toplam değeri
SQUAD_MIN_PLAYERS = 11             # değeri bilinen oyuncu sayısı bundan azsa -> NaN
VALUATION_MAX_AGE_DAYS = 540       # oyuncunun son değerlemesi bundan eskiyse sayılmaz
# Kadro değeri enflasyonunu gidermek için referans: 2012'den beri Transfermarkt'ta iç ligi olan ülkeler
SQUAD_REFERENCE_COUNTRIES = ("ENG", "ESP", "GER", "ITA", "FRA", "NED", "POR", "BEL",
                             "TUR", "GRE", "SCO", "RUS", "UKR", "DEN")

GOALS_FORM_WINDOW = 10             # atılan / yenen gol ortalaması için maç sayısı
GOALS_FORM_MIN_MATCHES = 5

PLAYER_WINDOW_MATCHES = 5          # "düzenli ilk 11" kulübün son kaç maçındaki dakikalardan seçilir
XI_SIZE = 11
BENCH_SIZE = 7                     # 12-18. sıradaki oyuncular
XI_MIN_VALUED = 8                  # ilk 11'de değeri bilinen oyuncu bundan azsa değer özellikleri NaN
BENCH_MIN_VALUED = 3
PLAYER_STATE_MAX_GAP_DAYS = 200    # kulübün son Transfermarkt maçı D'den bu kadar eskiyse NaN
VALUE_INDEX_WINDOW_DAYS = 365      # aylık değer indeksi penceresi
PRODUCTION_WINDOW_DAYS = 365       # gol+asist üretimi penceresi
PRODUCTION_MIN_MINUTES = 900
UEFA_EXPERIENCE_DAYS = 1095        # UEFA tecrübesi penceresi (3 yıl)
CONTINUITY_MIN_MINUTES = 45        # "son maçta oynadı" eşiği
REST_DAYS_CAP = 30                 # dinlenme süresi bu değerde kırpılır (sezon arası = 30)
UEFA_MAIN_COMPETITION_IDS = tuple(sorted(dc.UEFA_MAIN_COMPETITIONS))

FEATURE_PARAMS = {
    "ELO_BASE": ELO_BASE,
    "HOME_ELO_ADVANTAGE": HOME_ELO_ADVANTAGE,
    "ELO_MAX_STALENESS_DAYS": ELO_MAX_STALENESS_DAYS,
    "ELO_PROVISIONAL_MATCHES": ELO_PROVISIONAL_MATCHES,
    "ELO_PROVISIONAL_K_MULT": ELO_PROVISIONAL_K_MULT,
    "ELO_NEWCOMER_PERCENTILE": ELO_NEWCOMER_PERCENTILE,
    "ELO_NEWCOMER_MIN_POOL": ELO_NEWCOMER_MIN_POOL,
    "ELO_BURN_IN_PASSES": ELO_BURN_IN_PASSES,
    "ELO_BURN_IN_END": f"{dc.TRAIN_FIRST_SEASON_START}-07-01",
    "ELO_DATA_START": f"{dc.DEFAULT_FIRST_SEASON_START}-07-01",
    "FORM_WINDOW": FORM_WINDOW,
    "FORM_MIN_MATCHES": FORM_MIN_MATCHES,
    "FORM_MAX_GAP_DAYS": FORM_MAX_GAP_DAYS,
    "SQUAD_WINDOW_DAYS": SQUAD_WINDOW_DAYS,
    "SQUAD_TOP_N": SQUAD_TOP_N,
    "SQUAD_MIN_PLAYERS": SQUAD_MIN_PLAYERS,
    "VALUATION_MAX_AGE_DAYS": VALUATION_MAX_AGE_DAYS,
    "SQUAD_REFERENCE_COUNTRIES": list(SQUAD_REFERENCE_COUNTRIES),
    "POINTS": {"win": 3, "draw": 1, "loss": 0},
    "GOALS_FORM_WINDOW": GOALS_FORM_WINDOW,
    "GOALS_FORM_MIN_MATCHES": GOALS_FORM_MIN_MATCHES,
    "PLAYER_WINDOW_MATCHES": PLAYER_WINDOW_MATCHES,
    "XI_SIZE": XI_SIZE,
    "BENCH_SIZE": BENCH_SIZE,
    "XI_MIN_VALUED": XI_MIN_VALUED,
    "BENCH_MIN_VALUED": BENCH_MIN_VALUED,
    "PLAYER_STATE_MAX_GAP_DAYS": PLAYER_STATE_MAX_GAP_DAYS,
    "VALUE_INDEX_WINDOW_DAYS": VALUE_INDEX_WINDOW_DAYS,
    "PRODUCTION_WINDOW_DAYS": PRODUCTION_WINDOW_DAYS,
    "PRODUCTION_MIN_MINUTES": PRODUCTION_MIN_MINUTES,
    "UEFA_EXPERIENCE_DAYS": UEFA_EXPERIENCE_DAYS,
    "CONTINUITY_MIN_MINUTES": CONTINUITY_MIN_MINUTES,
    "REST_DAYS_CAP": REST_DAYS_CAP,
    "UEFA_MAIN_COMPETITION_IDS": list(UEFA_MAIN_COMPETITION_IDS),
    "ODDS_PRIORITY": list(dc.ODDS_PRIORITY),
    "SHOTS_WINDOW": SHOTS_WINDOW,
    "SHOTS_MIN_MATCHES": SHOTS_MIN_MATCHES,
    "SECOND_TIER_LEAGUES": sorted(dc.SECOND_TIER_LEAGUES),
}

# Model çıktısının sınıf sırası (predict_proba kolonları bu sırada)
CLASS_LABELS = ["home_win", "draw", "away_win"]
RESULT_TO_CLASS = {"H": 0, "D": 1, "A": 2}

_MATCH_SET = (
    "Maç kümesi: football-data.co.uk'teki 21 Avrupa 1. ligi + 6 adet 2. lig "
    f"({sorted(dc.SECOND_TIER_LEAGUES)}) + Transfermarkt'taki Ukrayna, Hırvatistan, Çekya, Sırbistan "
    "ligleri + UEFA Şampiyonlar Ligi / Avrupa Ligi / Konferans Ligi (ön eleme turları dahil). "
    "İç kupalar dahil DEĞİL."
)

FEATURE_SPECS = [
    {
        "name": "elo_diff",
        "dtype": "float64",
        "source": "Kendi ligler arası Elo (football-data + Transfermarkt maç sonuçları)",
        "definition": (
            f"elo(ev, D) + {HOME_ELO_ADVANTAGE} * (1 - neutral) - elo(deplasman, D). "
            "elo(kulüp, D): kulübün tarihi D'den KESİN OLARAK önceki son maçından sonraki Elo "
            f"rating'i (son maç D'den {ELO_MAX_STALENESS_DAYS} günden eskiyse NaN). {_MATCH_SET} "
            f"Elo motoru: veri {FEATURE_PARAMS['ELO_DATA_START']}'de başlar, maçlar gün gün işlenir "
            "(aynı gündeki tüm maçlar gün başındaki ratinglerle hesaplanır, güncellemeler gün sonunda "
            "uygulanır). Beklenen skor E = 1 / (1 + 10^(-(R_ev - R_dep + "
            f"{HOME_ELO_ADVANTAGE}*(1-neutral)) / 400)); S = 1 / 0.5 / 0. "
            "Değişim = K * u * p * G * (S - E); u = uefa_k_mult (UEFA maçıysa) yoksa 1; "
            f"p = {ELO_PROVISIONAL_K_MULT} (kulübün ilk {ELO_PROVISIONAL_MATCHES} maçı) yoksa 1 (her kulüp "
            "için ayrı); G = 1 (|gol farkı| <= 1), 1.5 (= 2), (11 + |gol farkı|) / 8 (>= 3). "
            "K ve uefa_k_mult eğitimde ayarlanır ve şemadaki elo_params altında yazılıdır. "
            f"Yeni kulüp başlangıcı: aynı ülkedeki mevcut ratinglerin %{ELO_NEWCOMER_PERCENTILE} "
            "yüzdeliği (ülkesi bilinmeyen kulüpte ülkesi bilinmeyen kulüplerin havuzu); havuzda "
            f"{ELO_NEWCOMER_MIN_POOL}'dan az kulüp varsa {ELO_BASE}. Isınma: "
            f"{FEATURE_PARAMS['ELO_DATA_START']} - {FEATURE_PARAMS['ELO_BURN_IN_END']} arası maçlar "
            f"{ELO_BURN_IN_PASSES} kez oynatılır (her tur bir öncekinin son ratingleriyle başlar), "
            "sonra tüm veri son kez işlenir. neutral: tarafsız saha ise 1."
        ),
    },
    {
        "name": "home_form",
        "dtype": "float64",
        "source": "football-data.co.uk + Transfermarkt maç sonuçları",
        "definition": (
            f"Ev sahibi takımın tarihi D'den KESİN OLARAK ÖNCEKİ son {FORM_WINDOW} maçındaki puan "
            "ortalaması (galibiyet 3, beraberlik 1, mağlubiyet 0; ev + deplasman; lig ve UEFA maçları, "
            "maç kümesi elo_diff ile aynı). Eğitimde takım bazında "
            f"points.shift(1).rolling({FORM_WINDOW}, min_periods={FORM_MIN_MATCHES}).mean() ile "
            f"birebir aynıdır. Pencerede {FORM_MIN_MATCHES}'ten az maç varsa NaN. Ardışık iki maç "
            f"arasında {FORM_MAX_GAP_DAYS} günden uzun boşluk varsa pencere sıfırlanır; son maç D'den "
            f"{FORM_MAX_GAP_DAYS} günden eskiyse NaN. Değer aralığı 0-3."
        ),
    },
    {
        "name": "away_form",
        "dtype": "float64",
        "source": "football-data.co.uk + Transfermarkt maç sonuçları",
        "definition": "home_form ile aynı hesaplama, deplasman takımı için.",
    },
    {
        "name": "home_squad_value",
        "dtype": "float64",
        "source": "Transfermarkt appearances + player_valuations",
        "definition": (
            "M = D'nin bulunduğu ayın ilk günü. [M - "
            f"{SQUAD_WINDOW_DAYS} gün, M) aralığında kulüp için Transfermarkt'ta maç kaydı olan "
            "oyuncular alınır; her oyuncunun tarihi M'den KESİN OLARAK önceki son piyasa değeri "
            f"(değerleme M'den {VALUATION_MAX_AGE_DAYS} günden eskiyse sayılmaz) kullanılır. "
            f"raw = log10(en değerli {SQUAD_TOP_N} oyuncunun toplam değeri, EUR); değeri bilinen oyuncu "
            f"sayısı {SQUAD_MIN_PLAYERS}'den azsa NaN. Sonuç = raw - aynı M için, iç ligi "
            f"{list(SQUAD_REFERENCE_COUNTRIES)} ülkelerinden olan kulüplerin raw medyanı "
            "(piyasa değeri enflasyonunu giderir). Transfermarkt kaydı olmayan kulüplerde NaN."
        ),
    },
    {
        "name": "away_squad_value",
        "dtype": "float64",
        "source": "Transfermarkt appearances + player_valuations",
        "definition": "home_squad_value ile aynı hesaplama, deplasman takımı için.",
    },
    {
        "name": "is_knockout",
        "dtype": "float64",
        "source": "Transfermarkt games.round",
        "definition": (
            "UEFA ana turnuvasında (CL/EL/UECL) grup/lig aşamasından SONRAKİ eleme turundaysa 1.0 "
            "(play-off/intermediate stage, son 16, çeyrek final, yarı final, final); grup/lig aşaması, "
            "ön eleme turları ve iç lig maçlarında 0.0."
        ),
    },
    {
        # Kullanıcının ilk özellik listesinde yoktu; doğrulama setinde (2024-25) UEFA maçlarının
        # log-loss'unu ve beraberlik kalibrasyonunu iyileştirdiği için eklendi (bkz. README).
        "name": "is_uefa",
        "dtype": "float64",
        "source": "maç kaynağı (Transfermarkt competition_id)",
        "definition": (
            "Maç UEFA Şampiyonlar Ligi / Avrupa Ligi / Konferans Ligi maçıysa (ön eleme turları dahil) "
            "1.0, iç lig maçıysa 0.0."
        ),
    },
]

_PLAYER_STATE = (
    "Oyuncu durumu (ortak tanım): T = kulübün tarihi D'den KESİN OLARAK önceki son Transfermarkt maçının "
    "tarihi (appearances; iç lig, iç kupa ve UEFA maçları). D - T > "
    f"{PLAYER_STATE_MAX_GAP_DAYS} gün ise tüm oyuncu özellikleri NaN. 'Düzenli ilk 11' = kulübün T dahil son "
    f"{PLAYER_WINDOW_MATCHES} maçında toplam en çok dakika alan {XI_SIZE} oyuncu (eşitlikte küçük player_id); "
    f"'yedekler' = aynı sıralamada {XI_SIZE + 1}-{XI_SIZE + BENCH_SIZE}. oyuncular. v(oyuncu) = tarihi <= T olan "
    f"son Transfermarkt piyasa değeri (T'den {VALUATION_MAX_AGE_DAYS} günden eskiyse değer yok sayılır). "
    "idx(T) = M (T'nin ayının ilk günü) için, [M - "
    f"{VALUE_INDEX_WINDOW_DAYS} gün, M) aralığında iç ligi {list(SQUAD_REFERENCE_COUNTRIES)} ülkelerinden "
    "olan kulüplerde forma giyen oyuncuların M'den KESİN OLARAK önceki son değerlerinin log10 medyanı "
    "(piyasa değeri enflasyonu düzeltmesi)."
)


def _side_pair(name: str, source: str, definition: str) -> list[dict]:
    return [
        {"name": f"home_{name}", "dtype": "float64", "source": source, "definition": definition},
        {"name": f"away_{name}", "dtype": "float64", "source": source,
         "definition": f"home_{name} ile aynı hesaplama, deplasman takımı için."},
    ]


_TM_PLAYERS = "Transfermarkt appearances + player_valuations + players"
FEATURE_SPECS += [
    *_side_pair("xi_value", _TM_PLAYERS,
                f"log10(düzenli ilk 11'de değeri bilinen oyuncuların v toplamı, EUR) - idx(T); değeri bilinen "
                f"oyuncu {XI_MIN_VALUED}'den azsa NaN. {_PLAYER_STATE}"),
    {"name": "xi_value_diff", "dtype": "float64", "source": _TM_PLAYERS,
     "definition": "home_xi_value - away_xi_value (idx(T) iki takımda aynı ay ise birbirini götürür)."},
    *_side_pair("xi_star", _TM_PLAYERS,
                f"log10(düzenli ilk 11'deki en yüksek v) - idx(T); değeri bilinen oyuncu {XI_MIN_VALUED}'den "
                "azsa NaN. (Oyuncu durumu tanımı: home_xi_value.)"),
    *_side_pair("bench_value", _TM_PLAYERS,
                f"log10(yedeklerin v toplamı) - idx(T); değeri bilinen yedek {BENCH_MIN_VALUED}'ten azsa NaN. "
                "(Oyuncu durumu tanımı: home_xi_value.)"),
    *_side_pair("att_value", _TM_PLAYERS,
                "log10(düzenli ilk 11'de players.position == 'Attack' olan oyuncuların v toplamı) - idx(T); "
                "bu mevkide değeri bilinen oyuncu yoksa NaN. (Oyuncu durumu tanımı: home_xi_value.)"),
    *_side_pair("mid_value", _TM_PLAYERS,
                "home_att_value ile aynı, players.position == 'Midfield' için."),
    *_side_pair("def_value", _TM_PLAYERS,
                "home_att_value ile aynı, players.position == 'Defender' için."),
    *_side_pair("gk_value", _TM_PLAYERS,
                "home_att_value ile aynı, players.position == 'Goalkeeper' için."),
    *_side_pair("xi_ga90", _TM_PLAYERS,
                "Düzenli ilk 11 oyuncularının (T - "
                f"{PRODUCTION_WINDOW_DAYS} gün, T] aralığındaki tüm Transfermarkt maçlarında: "
                "90 * toplam(gol + asist) / toplam(dakika). Toplam dakika "
                f"{PRODUCTION_MIN_MINUTES}'den azsa NaN. (Oyuncu durumu tanımı: home_xi_value.)"),
    *_side_pair("xi_age", _TM_PLAYERS,
                "Düzenli ilk 11 oyuncularının T tarihindeki yaş ortalaması (yıl = gün / 365.25; doğum tarihi "
                "bilinmeyenler hariç). (Oyuncu durumu tanımı: home_xi_value.)"),
    *_side_pair("xi_uefa_apps", _TM_PLAYERS,
                "Düzenli ilk 11 oyuncularının (T - "
                f"{UEFA_EXPERIENCE_DAYS} gün, T] aralığındaki UEFA ana turnuva ({list(UEFA_MAIN_COMPETITION_IDS)}; "
                "ön elemeler hariç) maç sayısı ortalaması (hangi kulüpte oynadığından bağımsız). "
                "(Oyuncu durumu tanımı: home_xi_value.)"),
    *_side_pair("xi_last_match_share", _TM_PLAYERS,
                "Düzenli ilk 11'de değeri bilinen oyuncuların v toplamı içinde, T tarihli maçta en az "
                f"{CONTINUITY_MIN_MINUTES} dakika oynayanların payı (0-1; rotasyon/sakatlık göstergesi). "
                "(Oyuncu durumu tanımı: home_xi_value.)"),
    *_side_pair("goals_for", "football-data.co.uk + Transfermarkt maç sonuçları",
                f"Takımın tarihi D'den KESİN OLARAK önceki son {GOALS_FORM_WINDOW} maçında attığı gol ortalaması "
                f"(maç kümesi elo_diff ile aynı; shift(1).rolling({GOALS_FORM_WINDOW}, "
                f"min_periods={GOALS_FORM_MIN_MATCHES}); pencere sıfırlama ve bayatlık kuralı home_form ile aynı)."),
    *_side_pair("goals_against", "football-data.co.uk + Transfermarkt maç sonuçları",
                "home_goals_for ile aynı, takımın YEDİĞİ goller için."),
    {"name": "mkt_elo_diff", "dtype": "float64",
     "source": "Piyasa Elo'su (football-data kapanış oranları + tüm maç sonuçları)",
     "definition": (
         f"mkt(ev, D) + {HOME_ELO_ADVANTAGE} * (1 - neutral) - mkt(deplasman, D). Piyasa Elo motoru elo_diff ile "
         "BİREBİR aynıdır (maç kümesi, gün bazlı güncelleme, ısınma, yeni kulüp, geçici K, bayatlık), tek fark "
         "değişim formülüdür: oranı olan maçlarda Değişim = K_m * u * p * [w * (E_piyasa - E) + (1 - w) * G * (S - E)], "
         "oranı olmayan maçlarda (UEFA, Transfermarkt ligleri) K_m * u * p * G * (S - E). "
         "E_piyasa = p_ev + 0.5 * p_beraberlik; p = (1/oran) / toplam(1/oran) (marj normalize). Oran: "
         f"{list(dc.ODDS_PRIORITY)} sırasındaki ilk geçerli üçlü (kapanış oranları önce). "
         "K_m (mkt_k), w (mkt_w) eğitimde ayarlanır, u = uefa_k_mult (elo_diff ile aynı); şemadaki elo_params. "
         "Oranlar yalnızca maçtan SONRA rating güncellemek için kullanılır; özellik D'den önceki ratingdir."
     )},
    *_side_pair("sot_share", "football-data.co.uk şut istatistikleri",
                f"Takımın tarihi D'den KESİN OLARAK önceki, şut verisi olan son {SHOTS_WINDOW} maçında "
                "toplam(isabetli şut) / (toplam(isabetli şut) + toplam(rakibin isabetli şutu)). Şut verili maç "
                f"sayısı {SHOTS_MIN_MATCHES}'ten azsa NaN; pencere sıfırlama ve bayatlık kuralı home_form ile aynı "
                "(boşluk ve bayatlık yalnızca şut verili maçlar üzerinden). UEFA maçlarında şut verisi yoktur."),
    *_side_pair("shot_share", "football-data.co.uk şut istatistikleri",
                "home_sot_share ile aynı, tüm şutlar (HS/AS) için."),
    *_side_pair("rest_days", "football-data.co.uk + Transfermarkt maç sonuçları ve appearances",
                "min(" f"{REST_DAYS_CAP}, D - takımın tarihi D'den KESİN OLARAK önceki son maçının tarihi). "
                "Son maç: elo_diff maç kümesi ile Transfermarkt appearances'taki (iç kupalar dahil) kulüp "
                "maçlarının birleşimi. Geçmiş maç yoksa NaN."),
]

# Özellik grupları: train.py 'base' grubunu her zaman kullanır, diğerlerini doğrulama setinde
# ileri seçimle ekler.
FEATURE_GROUPS = {
    "base": ["elo_diff", "is_knockout", "is_uefa"],
    "market": ["mkt_elo_diff"],
    "shots": ["home_sot_share", "away_sot_share", "home_shot_share", "away_shot_share"],
    "form": ["home_form", "away_form"],
    "squad_value": ["home_squad_value", "away_squad_value"],
    "xi_value": ["home_xi_value", "away_xi_value", "xi_value_diff", "home_xi_star", "away_xi_star",
                 "home_bench_value", "away_bench_value"],
    "lines": ["home_att_value", "away_att_value", "home_mid_value", "away_mid_value",
              "home_def_value", "away_def_value", "home_gk_value", "away_gk_value"],
    "production": ["home_xi_ga90", "away_xi_ga90"],
    "goals_form": ["home_goals_for", "away_goals_for", "home_goals_against", "away_goals_against"],
    "rest": ["home_rest_days", "away_rest_days"],
    "experience": ["home_xi_age", "away_xi_age", "home_xi_uefa_apps", "away_xi_uefa_apps"],
    "continuity": ["home_xi_last_match_share", "away_xi_last_match_share"],
}
_group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
for _spec in FEATURE_SPECS:
    _spec["group"] = _group_of[_spec["name"]]
FEATURE_COLUMNS = [s["name"] for s in FEATURE_SPECS]
assert sorted(FEATURE_COLUMNS) == sorted(_group_of), "FEATURE_GROUPS ile FEATURE_SPECS uyuşmuyor"


def schema_contract(feature_columns: list[str], elo_params: dict, ensemble: dict) -> dict:
    """
    Modelin girdi/çıktı sözleşmesi: kullanılan özellikler (sırasıyla), tanımları, sabit parametreler,
    eğitimde ayarlanan Elo parametreleri, topluluk parametreleri (ağırlıklar, Dixon-Coles rho,
    sıcaklık) ve sınıf sırası. Biri değişirse hash değişir.
    """
    specs = {s["name"]: s for s in FEATURE_SPECS}
    unknown = [c for c in feature_columns if c not in specs]
    if unknown:
        raise ValueError(f"Tanımsız özellik(ler): {unknown}")
    return {
        "feature_columns": list(feature_columns),
        "features": [specs[c] for c in feature_columns],
        "params": FEATURE_PARAMS,
        "elo_params": {k: float(v) for k, v in elo_params.items()},
        "ensemble": {
            "weights": {k: float(v) for k, v in ensemble["weights"].items()},
            "rho": float(ensemble["rho"]),
            "temperature": float(ensemble["temperature"]),
            "max_goals": int(ensemble["max_goals"]),
        },
        "class_order": CLASS_LABELS,
    }


def contract_hash(contract: dict) -> str:
    payload = json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# Takım adı normalizasyonu
# =============================================================================

TEAM_ALIASES_PATH = dc.DATA_DIR / "team_aliases.csv"

# Anlamsız ekler: "FC Porto" ile "Porto" aynı normalize ada düşsün
_STOP_TOKENS = {
    "fc", "cf", "afc", "cfc", "sc", "ac", "as", "ss", "ssc", "sv", "vfb", "vfl", "tsg",
    "fk", "sk", "bk", "nk", "kv", "krc", "club", "calcio", "cd", "ud", "sad", "the",
    "if", "ik", "ff",  # İskandinav kulüp ekleri ("Östers IF" -> "osters")
}
# Kısaltma açılımları; iki tarafa da uygulandığı için simetriktir.
# ("ath" bilerek yok: football-data hem Athletic Bilbao hem Atletico Madrid için "Ath" kullanıyor.)
_TOKEN_EXPANSIONS = {"man": "manchester", "utd": "united", "st": "saint", "nottm": "nottingham"}


@lru_cache(maxsize=None)
def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    # NFKD'nin ayrıştırmadığı harfler
    text = text.replace("ø", "o").replace("æ", "ae").replace("ß", "ss").replace("ł", "l").replace("'", "")
    tokens = re.split(r"[^a-z0-9]+", text)
    tokens = [_TOKEN_EXPANSIONS.get(t, t) for t in tokens if t and t not in _STOP_TOKENS]
    return "".join(tokens)


def _unique_name_match(name: str, pool: dict[str, str]) -> tuple[str | None, str]:
    """
    pool: anahtar -> görünen ad. Sırasıyla normalize eşitlik, önek (>=3 karakter),
    alt-dizi (>=5 karakter), difflib >= 0.88. Her adımda TEK aday şartı aranır.
    """
    n = normalize_team_name(name)
    if not n:
        return None, "unresolved"
    norm = {k: normalize_team_name(v) for k, v in pool.items()}
    steps = [
        ("normalized", lambda m: m == n),
        ("prefix", lambda m: len(min(m, n, key=len)) >= 3 and (m.startswith(n) or n.startswith(m))),
        ("substring", lambda m: len(min(m, n, key=len)) >= 5 and (m in n or n in m)),
    ]
    for method, pred in steps:
        hits = {k for k, m in norm.items() if m and pred(m)}
        if len(hits) == 1:
            return hits.pop(), method
        if len(hits) > 1:
            return None, f"ambiguous_{method}"
    close = difflib.get_close_matches(n, list(set(norm.values())), n=2, cutoff=0.88)
    if len(close) == 1:
        hits = [k for k, m in norm.items() if m == close[0]]
        if len(hits) == 1:
            return hits[0], "fuzzy"
    return None, "unresolved"


# =============================================================================
# Takım kimliği
# =============================================================================

class TeamIdentity:
    """Kanonik anahtarlar, görünen adlar, ülkeler ve ad -> anahtar çözümlemesi."""

    def __init__(self, fd_keys: dict, display: dict, country: dict, names: dict, report: pd.DataFrame):
        self.fd_keys = fd_keys          # (ülke, football-data adı) -> anahtar
        self.display = display          # anahtar -> görünen ad
        self.country = country          # anahtar -> ülke kodu (bilinmiyorsa yok)
        self.names = names              # anahtar -> bilinen tüm adlar
        self.report = report
        self._lower: dict[str, set] = defaultdict(set)
        self._norm: dict[str, set] = defaultdict(set)
        for key, aliases in names.items():
            for a in aliases:
                self._lower[str(a).strip().lower()].add(key)
                self._norm[normalize_team_name(a)].add(key)

    def resolve(self, name: str) -> tuple[str | None, str | list]:
        """(anahtar, yöntem) ya da (None, aday listesi) döner."""
        name = str(name).strip()
        if name in self.display:
            return name, "key"
        for index, value, method in ((self._lower, name.lower(), "exact"),
                                     (self._norm, normalize_team_name(name), "normalized")):
            hits = index.get(value, set())
            if len(hits) == 1:
                return next(iter(hits)), method
            if len(hits) > 1:
                return None, sorted(f"{self.display[k]} ({k})" for k in hits)
        key, method = _unique_name_match(name, self.display)
        if key:
            return key, method
        # Çözülemedi: en benzer 5 kulübü öneri olarak döndür
        ranked = sorted(self.display.items(),
                        key=lambda kv: -difflib.SequenceMatcher(None, name.lower(), kv[1].lower()).ratio())
        return None, [f"{v} ({k})" for k, v in ranked[:5]]


def _load_aliases() -> pd.DataFrame:
    if not TEAM_ALIASES_PATH.exists():
        return pd.DataFrame(columns=["alias", "target", "country"])
    al = pd.read_csv(TEAM_ALIASES_PATH, comment="#", dtype=str)
    for c in ("alias", "target", "country"):
        if c not in al.columns:
            al[c] = None
    return al.dropna(subset=["alias", "target"])


def build_team_identity(fd: pd.DataFrame, tm: pd.DataFrame, tm_clubs: pd.DataFrame) -> TeamIdentity:
    # --- Transfermarkt kulüpleri: görünen ad, bilinen adlar, ülke ---
    tm_long = pd.concat([
        tm[["home_tm_id", "home_team", "date"]].set_axis(["tm_id", "name", "date"], axis=1),
        tm[["away_tm_id", "away_team", "date"]].set_axis(["tm_id", "name", "date"], axis=1),
    ]).dropna(subset=["tm_id"])
    tm_long["tm_id"] = tm_long["tm_id"].astype(int)
    latest_name = tm_long.sort_values("date").groupby("tm_id")["name"].last()
    display = {f"tm:{i}": n for i, n in latest_name.items()}
    display.update({f"tm:{i}": n for i, n in zip(tm_clubs["club_id"], tm_clubs["name"])})
    names = defaultdict(set)
    for i, n in zip(tm_long["tm_id"], tm_long["name"]):
        names[f"tm:{i}"].add(n)
    for i, n in zip(tm_clubs["club_id"], tm_clubs["name"]):
        names[f"tm:{i}"].add(n)
    country = {f"tm:{i}": c for i, c in zip(tm_clubs["club_id"], tm_clubs["country"]) if isinstance(c, str)}
    dom = tm[~tm["is_uefa"]]
    for col in ("home_tm_id", "away_tm_id"):
        for i, c in zip(dom[col], dom["country"]):
            country.setdefault(f"tm:{int(i)}", c)

    # --- 1) aynı gün + aynı skor çakıştırması (co-occurrence) ---
    fd_names = pd.concat([fd[["country", "home_team"]].set_axis(["country", "name"], axis=1),
                          fd[["country", "away_team"]].set_axis(["country", "name"], axis=1)]).drop_duplicates()
    # Yalnızca 1. lig maçları: Transfermarkt'ta 2. lig yok; 2. lig maçlarını dahil etmek aynı gün aynı skorlu
    # 1. lig maçlarıyla tesadüfi eşleşme üretir (ör. "Pordenone" -> Hellas Verona). 2. ligde oynayan eski
    # 1. lig takımları (ülke, ad) anahtarı ortak olduğu için yine eşlenir.
    joined = fd.loc[fd["tier"] == 1, ["country", "date", "home_goals", "away_goals", "home_team", "away_team"]].merge(
        dom[["country", "date", "home_goals", "away_goals", "home_tm_id", "away_tm_id"]],
        on=["country", "date", "home_goals", "away_goals"])
    pairs = pd.concat([
        joined[["country", "home_team", "home_tm_id"]].set_axis(["country", "name", "tm_id"], axis=1),
        joined[["country", "away_team", "away_tm_id"]].set_axis(["country", "name", "tm_id"], axis=1),
    ])
    counts = pairs.value_counts().reset_index(name="n").sort_values("n", ascending=False)
    fd_keys: dict[tuple, str] = {}
    rows = []
    for (c, name), grp in counts.groupby(["country", "name"], sort=False):
        n1 = grp["n"].iloc[0]
        n2 = grp["n"].iloc[1] if len(grp) > 1 else 0
        if n1 >= 3 and n1 >= 3 * n2:
            key = f"tm:{int(grp['tm_id'].iloc[0])}"
            fd_keys[(c, name)] = key
            rows.append((c, name, key, "cooccurrence", int(n1)))

    # --- 2) data/team_aliases.csv ---
    display_lookup = defaultdict(set)
    for k, n in display.items():
        display_lookup[n.lower()].add(k)
    aliases = _load_aliases()
    for alias, target, al_country in zip(aliases["alias"], aliases["target"], aliases["country"]):
        key = target if target.startswith(("tm:", "fd:")) else (
            next(iter(display_lookup[target.lower()])) if len(display_lookup[target.lower()]) == 1 else None)
        if key is None:
            log.warning("team_aliases.csv: hedef bulunamadı ya da belirsiz: %s -> %s", alias, target)
            continue
        names[key].add(alias)
        for c, name in fd_names.itertuples(index=False):
            if name == alias and (not isinstance(al_country, str) or al_country == c):
                fd_keys[(c, name)] = key
                rows = [r for r in rows if (r[0], r[1]) != (c, name)]
                rows.append((c, name, key, "alias", 0))

    # --- 3) kalan adlar: aynı ülkenin ya da ülkesi bilinmeyen TM kulüpleri arasında ad eşleştirmesi ---
    claimed = defaultdict(set)
    for (c, _), key in fd_keys.items():
        claimed[c].add(key)
    candidates = []  # (ülke, ad, anahtar, yöntem)
    for c, name in fd_names.itertuples(index=False):
        if (c, name) in fd_keys:
            continue
        # İç ligi Transfermarkt'ta 2012'den beri olan ülkelerin kulüpleri Transfermarkt'ta ülkesiyle
        # birlikte vardır; bunlar için "ülkesi bilinmeyen" havuz (Cebelitarık, Meksika kulüpleri vb.)
        # kullanılmaz ("Lincoln" -> "Lincoln Red Imps" gibi hatalar).
        allowed = (c,) if c in SQUAD_REFERENCE_COUNTRIES else (c, None)
        pool = {k: v for k, v in display.items()
                if country.get(k) in allowed and k not in claimed[c]}
        key, method = _unique_name_match(name, pool)
        candidates.append((c, name, key, method))
    # Aynı ülkede normalize hali farklı iki ad aynı kulübe düştüyse yalnızca en güçlü
    # eşleşme kalır ("Osters" önekle "Östersunds FK"ya düşerken "Ostersunds" birebir eşleşir).
    rank = {"normalized": 0, "prefix": 1, "substring": 2, "fuzzy": 3}
    best: dict[tuple, tuple] = {}
    for c, name, key, method in candidates:
        if key is None:
            continue
        cur = best.get((c, key))
        score = (rank[method], name)
        if cur is None or score < cur:
            best[(c, key)] = score
    for c, name, key, method in candidates:
        winner = key is not None and best[(c, key)][1] == name
        same_norm = key is not None and normalize_team_name(best[(c, key)][1]) == normalize_team_name(name)
        if key is not None and (winner or same_norm):
            fd_keys[(c, name)] = key
            rows.append((c, name, key, f"name_{method}", 0))
        else:
            fd_keys[(c, name)] = f"fd:{c}:{normalize_team_name(name)}"
            rows.append((c, name, fd_keys[(c, name)], "conflict_lost" if key else method, 0))

    for (c, name), key in fd_keys.items():
        names[key].add(name)
        display.setdefault(key, name)
        country.setdefault(key, c)
    report = pd.DataFrame(rows, columns=["country", "fd_name", "key", "method", "cooccurrence_n"])
    report["key_display"] = report["key"].map(display)
    return TeamIdentity(fd_keys, display, country, dict(names), report)


def assemble_matches(fd: pd.DataFrame, tm: pd.DataFrame, identity: TeamIdentity) -> pd.DataFrame:
    """
    Tek maç tablosu: football-data iç ligleri + football-data'da olmayan ülkelerin Transfermarkt
    iç ligleri + Transfermarkt UEFA maçları. Aynı maç iki kaynaktan birden alınmaz.
    """
    f = fd.copy()
    f["home_key"] = [identity.fd_keys[(c, n)] for c, n in zip(f["country"], f["home_team"])]
    f["away_key"] = [identity.fd_keys[(c, n)] for c, n in zip(f["country"], f["away_team"])]
    t = tm[tm["is_uefa"] | ~tm["country"].isin(dc.FOOTBALL_DATA_COUNTRIES)].copy()
    t["home_key"] = "tm:" + t["home_tm_id"].astype(int).astype(str)
    t["away_key"] = "tm:" + t["away_tm_id"].astype(int).astype(str)
    m = pd.concat([f, t], ignore_index=True)
    return m.sort_values(["date", "home_key"], kind="stable").reset_index(drop=True)


def drop_same_day_duplicates(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Aynı kulübün aynı gün birden fazla maçı olamaz; varsa bu neredeyse her zaman
    yanlış kimlik eşleştirmesi ya da kaynak tekrarıdır. Bu maçlar tamamen çıkarılır.
    """
    both = pd.concat([matches[["home_key", "date"]].set_axis(["team", "date"], axis=1),
                      matches[["away_key", "date"]].set_axis(["team", "date"], axis=1)])
    bad = both[both.duplicated(keep=False)].drop_duplicates()
    if bad.empty:
        return matches
    log.warning("%d takım-gün çakışması — ilgili maçlar atıldı (ör. %s)",
                len(bad), bad["team"].unique()[:8].tolist())
    bad_keys = set(zip(bad["team"], bad["date"]))
    mask = np.array([(h, d) in bad_keys or (a, d) in bad_keys
                     for h, a, d in zip(matches["home_key"], matches["away_key"], matches["date"])])
    return matches[~mask].reset_index(drop=True)


# =============================================================================
# Ligler arası Elo
# =============================================================================

def _goal_diff_multiplier(gd: np.ndarray) -> np.ndarray:
    gd = np.abs(gd)
    return np.where(gd <= 1, 1.0, np.where(gd == 2, 1.5, (11.0 + gd) / 8.0))


def market_expected_score(matches: pd.DataFrame) -> np.ndarray:
    """Oranların ima ettiği ev sahibi beklenen skoru p_ev + 0.5 * p_beraberlik (oran yoksa NaN)."""
    odds = matches[["odds_home", "odds_draw", "odds_away"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / odds
        p = inv / inv.sum(axis=1, keepdims=True)
    e = p[:, 0] + 0.5 * p[:, 1]
    valid = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    return np.where(valid, e, np.nan)


def run_elo(matches: pd.DataFrame, k: float, uefa_k_mult: float, country_of: dict,
            record_history: bool = True, market_weight: float = 0.0) -> dict:
    """
    Elo motoru (tanım: FEATURE_SPECS['elo_diff']; market_weight > 0 ise 'mkt_elo_diff').
    market_weight = 0 iken klasik sonuç-tabanlı Elo ile birebir aynıdır.
    Dönüş: {'history': DataFrame(team, day, elo) — yalnızca son tur,
            'pre_home', 'pre_away': son turdaki maç öncesi ratingler}
    """
    day_arr = matches["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    burn_in_end = np.datetime64(FEATURE_PARAMS["ELO_BURN_IN_END"], "D").astype(np.int64)
    n_burn = int(np.searchsorted(day_arr, burn_in_end, side="left"))
    # Sıcak döngü düz Python listeleriyle çok daha hızlı (numpy skaler erişimi pahalı)
    day = day_arr.tolist()
    home = matches["home_key"].tolist()
    away = matches["away_key"].tolist()
    score = matches["result"].map({"H": 1.0, "D": 0.5, "A": 0.0}).tolist()
    gmult = _goal_diff_multiplier(matches["home_goals"].to_numpy() - matches["away_goals"].to_numpy()).tolist()
    kmult = (np.where(matches["is_uefa"].to_numpy(bool), uefa_k_mult, 1.0) * k).tolist()
    hfa = (HOME_ELO_ADVANTAGE * (1.0 - matches["neutral"].to_numpy(float))).tolist()
    if market_weight > 0:
        e_mkt = market_expected_score(matches)
        has_mkt = ~np.isnan(e_mkt)
        w_arr = np.where(has_mkt, market_weight, 0.0).tolist()
        e_mkt = np.where(has_mkt, e_mkt, 0.0).tolist()
    else:
        w_arr = [0.0] * len(matches)
        e_mkt = w_arr

    ratings: dict[str, float] = {}
    played: dict[str, int] = defaultdict(int)
    pool: dict[object, set] = defaultdict(set)  # ülke (None = bilinmiyor) -> rating'i olan kulüpler

    def init_rating(key):
        c = country_of.get(key)
        members = pool[c]
        if len(members) >= ELO_NEWCOMER_MIN_POOL:
            return float(np.percentile([ratings[m] for m in members], ELO_NEWCOMER_PERCENTILE))
        return ELO_BASE

    def one_pass(end: int, prior: dict, record: bool):
        ratings.clear()
        pool.clear()
        pre_h = [0.0] * end
        pre_a = [0.0] * end
        hist_team, hist_day, hist_elo = [], [], []
        i = 0
        while i < end:
            j = i
            d = day[i]
            while j < end and day[j] == d:
                j += 1
            for t in range(i, j):
                for key in (home[t], away[t]):
                    if key not in ratings:
                        ratings[key] = prior[key] if key in prior else init_rating(key)
                        pool[country_of.get(key)].add(key)
                pre_h[t] = ratings[home[t]]
                pre_a[t] = ratings[away[t]]
            delta = defaultdict(float)
            for t in range(i, j):
                h, a = home[t], away[t]
                e = 1.0 / (1.0 + 10.0 ** (-(pre_h[t] - pre_a[t] + hfa[t]) / 400.0))
                w = w_arr[t]
                change = kmult[t] * (w * (e_mkt[t] - e) + (1.0 - w) * gmult[t] * (score[t] - e))
                ph = ELO_PROVISIONAL_K_MULT if played[h] < ELO_PROVISIONAL_MATCHES else 1.0
                pa = ELO_PROVISIONAL_K_MULT if played[a] < ELO_PROVISIONAL_MATCHES else 1.0
                delta[h] += ph * change
                delta[a] -= pa * change
            for key, dv in delta.items():
                ratings[key] += dv
                played[key] += 1
                if record:
                    hist_team.append(key)
                    hist_day.append(d)
                    hist_elo.append(ratings[key])
            i = j
        return pre_h, pre_a, (hist_team, hist_day, hist_elo)

    prior: dict[str, float] = {}
    for _ in range(ELO_BURN_IN_PASSES):
        one_pass(n_burn, prior, record=False)
        prior = dict(ratings)
    pre_h, pre_a, (ht, hd, he) = one_pass(len(matches), prior, record=record_history)
    history = pd.DataFrame({"team": ht, "day": np.asarray(hd, dtype=np.int64), "elo": he})
    return {"history": history, "pre_home": np.asarray(pre_h), "pre_away": np.asarray(pre_a)}


def _elo_objective(matches: pd.DataFrame, res: dict, until: pd.Timestamp) -> tuple[float, float]:
    """
    Ayar ölçütü: [ısınma sonu, until) aralığındaki 1. lig ve UEFA maçlarında, Elo beklenen skorunun
    gerçek skora ortalama kare hatası (iç lig, UEFA). 2. lig maçları ölçüte girmez.
    """
    window = ((matches["date"] >= pd.Timestamp(FEATURE_PARAMS["ELO_BURN_IN_END"])) & (matches["date"] < until)
              & ((matches["tier"] == 1) | matches["is_uefa"])).to_numpy()
    uefa = matches["is_uefa"].to_numpy(bool)
    score = matches["result"].map({"H": 1.0, "D": 0.5, "A": 0.0}).to_numpy()
    hfa = HOME_ELO_ADVANTAGE * (1.0 - matches["neutral"].to_numpy(float))
    e = 1.0 / (1.0 + 10.0 ** (-(res["pre_home"] - res["pre_away"] + hfa) / 400.0))
    err = (e - score) ** 2
    return float(err[window & ~uefa].mean()), float(err[window & uefa].mean())


def tune_elo(matches: pd.DataFrame, country_of: dict, until: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """
    İki aşamalı ızgara araması, YALNIZCA until öncesi maçlarla (doğrulama/test dönemi kullanılmaz):
      1) Sonuç Elo'su: K, uefa_k_mult
      2) Piyasa Elo'su: mkt_k, mkt_w (uefa_k_mult 1. aşamadan)
    Ölçüt: iç lig ve UEFA ortalama kare hatalarının ortalaması (UEFA az ama ligler arası ölçeği o belirler).
    """
    rows = []
    for k in ELO_K_GRID:
        for mult in ELO_UEFA_K_MULT_GRID:
            mse_dom, mse_uefa = _elo_objective(matches, run_elo(matches, k, mult, country_of, record_history=False), until)
            rows.append({"engine": "elo", "k": k, "uefa_k_mult": mult, "market_weight": 0.0,
                         "mse_domestic": mse_dom, "mse_uefa": mse_uefa, "objective": 0.5 * (mse_dom + mse_uefa)})
            log.info("  elo K=%-4s uefa_mult=%-3s mse_dom=%.5f mse_uefa=%.5f", k, mult, mse_dom, mse_uefa)
    table = pd.DataFrame(rows)
    best = table.sort_values("objective").iloc[0]
    params = {"k": float(best["k"]), "uefa_k_mult": float(best["uefa_k_mult"])}

    for k in MARKET_K_GRID:
        for w in MARKET_WEIGHT_GRID:
            res = run_elo(matches, k, params["uefa_k_mult"], country_of, record_history=False, market_weight=w)
            mse_dom, mse_uefa = _elo_objective(matches, res, until)
            rows.append({"engine": "market", "k": k, "uefa_k_mult": params["uefa_k_mult"], "market_weight": w,
                         "mse_domestic": mse_dom, "mse_uefa": mse_uefa, "objective": 0.5 * (mse_dom + mse_uefa)})
            log.info("  piyasa K=%-4s w=%-4s mse_dom=%.5f mse_uefa=%.5f", k, w, mse_dom, mse_uefa)
    table = pd.DataFrame(rows)
    best_m = table[table["engine"] == "market"].sort_values("objective").iloc[0]
    params.update({"mkt_k": float(best_m["k"]), "mkt_w": float(best_m["market_weight"])})
    return params, table.sort_values(["engine", "objective"]).reset_index(drop=True)


# =============================================================================
# Form
# =============================================================================

def _to_days(values) -> np.ndarray:
    """Tarihleri 1970'ten itibaren gün sayısına (int64) çevirir."""
    return pd.to_datetime(pd.Series(values)).to_numpy().astype("datetime64[D]").astype(np.int64)


def _segment_rolling(df: pd.DataFrame, col: str, window: int, min_periods: int, shift: bool = False) -> pd.Series:
    """(team, segment) içinde kayan ortalama; shift=True ise mevcut satır hariç (sızıntı kontrolü için)."""
    s = df.groupby(["team", "segment"], sort=False)[col]
    if shift:
        s = s.shift(1).groupby([df["team"], df["segment"]], sort=False)
    return s.rolling(window, min_periods=min_periods).mean().reset_index(level=[0, 1], drop=True).sort_index()


def build_team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Takım-maç tablosu: team, date, day, points, goals_for, goals_against, segment ve bu maç DAHİL
    kayan ortalamalar (form_after, gf_after, ga_after). Sorgu tarafında strict (<) arama yapıldığı
    için "maç dahil" değer, bir sonraki maçın "maç öncesi" değeri olur (= shift(1)).
    """
    pts = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}
    home = pd.DataFrame({"team": matches["home_key"], "date": matches["date"],
                         "points": matches["result"].map(lambda r: pts[r][0]),
                         "goals_for": matches["home_goals"], "goals_against": matches["away_goals"]})
    away = pd.DataFrame({"team": matches["away_key"], "date": matches["date"],
                         "points": matches["result"].map(lambda r: pts[r][1]),
                         "goals_for": matches["away_goals"], "goals_against": matches["home_goals"]})
    log_df = pd.concat([home, away], ignore_index=True)
    log_df = log_df.sort_values(["team", "date"], kind="stable").reset_index(drop=True)
    log_df[["points", "goals_for", "goals_against"]] = log_df[["points", "goals_for", "goals_against"]].astype(float)
    log_df["day"] = _to_days(log_df["date"])
    gap = log_df.groupby("team")["day"].diff()
    log_df["segment"] = (gap.isna() | (gap > FORM_MAX_GAP_DAYS)).groupby(log_df["team"]).cumsum()
    log_df["form_after"] = _segment_rolling(log_df, "points", FORM_WINDOW, FORM_MIN_MATCHES)
    log_df["gf_after"] = _segment_rolling(log_df, "goals_for", GOALS_FORM_WINDOW, GOALS_FORM_MIN_MATCHES)
    log_df["ga_after"] = _segment_rolling(log_df, "goals_against", GOALS_FORM_WINDOW, GOALS_FORM_MIN_MATCHES)
    return log_df


def build_shots_log(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Şut verisi olan maçlardan takım tablosu ve bu maç DAHİL kayan paylar (sot_share_after,
    shot_share_after). Boşluk/segment kuralı yalnızca şut verili maçlar üzerinden işler.
    """
    m = matches.dropna(subset=["home_sot", "away_sot", "home_shots", "away_shots"])
    home = pd.DataFrame({"team": m["home_key"], "date": m["date"], "sot_for": m["home_sot"],
                         "sot_against": m["away_sot"], "shots_for": m["home_shots"], "shots_against": m["away_shots"]})
    away = pd.DataFrame({"team": m["away_key"], "date": m["date"], "sot_for": m["away_sot"],
                         "sot_against": m["home_sot"], "shots_for": m["away_shots"], "shots_against": m["home_shots"]})
    df = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable").reset_index(drop=True)
    df[["sot_for", "sot_against", "shots_for", "shots_against"]] = df[
        ["sot_for", "sot_against", "shots_for", "shots_against"]].astype(float)
    df["sot_total"] = df["sot_for"] + df["sot_against"]
    df["shots_total"] = df["shots_for"] + df["shots_against"]
    df["day"] = _to_days(df["date"])
    gap = df.groupby("team")["day"].diff()
    df["segment"] = (gap.isna() | (gap > FORM_MAX_GAP_DAYS)).groupby(df["team"]).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        for kind in ("sot", "shots"):
            num = _segment_rolling(df, f"{kind}_for", SHOTS_WINDOW, SHOTS_MIN_MATCHES)
            den = _segment_rolling(df, f"{kind}_total", SHOTS_WINDOW, SHOTS_MIN_MATCHES)
            name = "sot_share_after" if kind == "sot" else "shot_share_after"
            df[name] = (num / den).where(den > 0)
    return df


# =============================================================================
# Kadro piyasa değeri
# =============================================================================

def build_squad_value_table(appearances: pd.DataFrame, valuations: pd.DataFrame,
                            country_of: dict) -> pd.DataFrame:
    """Aylık anlık görüntüler: team, month (ay başı), day, squad_value (tanım: FEATURE_SPECS)."""
    ap = appearances.copy()
    ap["month"] = ap["date"].dt.to_period("M").dt.to_timestamp()
    # Aynı oyuncu-kulüp-ay için yalnızca o aydaki SON maç tarihi yeterli (pencere kontrolü için)
    ap = ap.groupby(["player_id", "club_id", "month"], as_index=False)["date"].max()
    parts = []
    for offset in range(1, SQUAD_WINDOW_DAYS // 28 + 2):
        snap = ap["month"] + pd.DateOffset(months=offset)
        ok = (snap - ap["date"]).dt.days <= SQUAD_WINDOW_DAYS
        parts.append(pd.DataFrame({"player_id": ap.loc[ok, "player_id"], "club_id": ap.loc[ok, "club_id"],
                                   "snapshot": snap[ok]}))
    roster = pd.concat(parts, ignore_index=True).drop_duplicates()

    val = valuations.sort_values("date")
    roster = roster.sort_values("snapshot")
    merged = pd.merge_asof(roster, val.rename(columns={"date": "valued_at"}),
                           left_on="snapshot", right_on="valued_at", by="player_id",
                           direction="backward", allow_exact_matches=False)
    merged = merged.dropna(subset=["value_eur"])
    merged = merged[(merged["snapshot"] - merged["valued_at"]).dt.days <= VALUATION_MAX_AGE_DAYS]

    merged = merged.sort_values("value_eur", ascending=False)
    top = merged.groupby(["club_id", "snapshot"], sort=False).head(SQUAD_TOP_N)
    agg = top.groupby(["club_id", "snapshot"]).agg(total=("value_eur", "sum"), n=("value_eur", "size"))
    counts = merged.groupby(["club_id", "snapshot"]).size()
    agg = agg[counts.reindex(agg.index) >= SQUAD_MIN_PLAYERS].reset_index()
    agg["team"] = "tm:" + agg["club_id"].astype(int).astype(str)
    agg["raw"] = np.log10(agg["total"])
    ref_mask = agg["team"].map(country_of).isin(SQUAD_REFERENCE_COUNTRIES)
    reference = agg[ref_mask].groupby("snapshot")["raw"].median()
    agg["squad_value"] = agg["raw"] - agg["snapshot"].map(reference)
    agg = agg.dropna(subset=["squad_value"])
    agg["day"] = _to_days(agg["snapshot"])
    return agg[["team", "snapshot", "day", "squad_value"]].reset_index(drop=True)


# =============================================================================
# Oyuncu bazlı takım gücü
# =============================================================================

def _month_start(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp()


def build_value_index(appearances: pd.DataFrame, valuations: pd.DataFrame, country_of: dict) -> pd.Series:
    """Aylık değer indeksi idx: ay başı -> log10 medyan oyuncu değeri (tanım: _PLAYER_STATE)."""
    clubs = pd.Series(appearances["club_id"].unique())
    ref_clubs = set(clubs[clubs.map(lambda c: country_of.get(f"tm:{c}") in SQUAD_REFERENCE_COUNTRIES)])
    ap = appearances.loc[appearances["club_id"].isin(ref_clubs), ["player_id", "date"]].copy()
    ap["month"] = _month_start(ap["date"])
    ap = ap.groupby(["player_id", "month"], as_index=False)["date"].max()
    parts = []
    for offset in range(1, VALUE_INDEX_WINDOW_DAYS // 28 + 2):
        snap = ap["month"] + pd.DateOffset(months=offset)
        ok = (snap - ap["date"]).dt.days <= VALUE_INDEX_WINDOW_DAYS
        parts.append(pd.DataFrame({"player_id": ap.loc[ok, "player_id"].to_numpy(), "month": snap[ok].to_numpy()}))
    roster = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values("month")
    val = valuations.rename(columns={"date": "valued_at"}).sort_values("valued_at")
    merged = pd.merge_asof(roster, val, left_on="month", right_on="valued_at", by="player_id",
                           direction="backward", allow_exact_matches=False)
    merged = merged.dropna(subset=["value_eur"])
    merged = merged[(merged["month"] - merged["valued_at"]).dt.days <= VALUATION_MAX_AGE_DAYS]
    return np.log10(merged["value_eur"]).groupby(merged["month"]).median()


def build_player_team_state(appearances: pd.DataFrame, valuations: pd.DataFrame, players: pd.DataFrame,
                            value_index: pd.Series) -> pd.DataFrame:
    """
    Kulübün her Transfermarkt maçından SONRAKİ oyuncu durumu (tanım: _PLAYER_STATE).
    Dönüş: team, date (T), day ve PLAYER_STATE_COLUMNS kolonları. Sorgu strict (<) yapılır.
    """
    ap = appearances
    # 1) Kulübün maç sırası
    cg = (ap[["club_id", "game_id", "date"]].drop_duplicates(["club_id", "game_id"])
          .sort_values(["club_id", "date", "game_id"], kind="stable").reset_index(drop=True))
    cg["idx"] = cg.groupby("club_id").cumcount()
    ap = ap.merge(cg[["club_id", "game_id", "idx"]], on=["club_id", "game_id"])
    last_idx = cg.groupby("club_id")["idx"].max()

    # 2) Son PLAYER_WINDOW_MATCHES maçın dakikaları -> sıralama (ilk 11 + yedekler)
    club = ap["club_id"].to_numpy(np.int64)
    parts = [pd.DataFrame({"club_id": club, "tidx": ap["idx"].to_numpy(np.int32) + o,
                           "player_id": ap["player_id"].to_numpy(np.int64),
                           "minutes": ap["minutes"].to_numpy(np.int32)})
             for o in range(PLAYER_WINDOW_MATCHES)]
    win = pd.concat(parts, ignore_index=True)
    win = win[win["tidx"].to_numpy() <= win["club_id"].map(last_idx).to_numpy()]
    win = win.groupby(["club_id", "tidx", "player_id"], as_index=False, sort=False)["minutes"].sum()
    win = win.sort_values(["club_id", "tidx", "minutes", "player_id"],
                          ascending=[True, True, False, True], kind="stable")
    win["rank"] = win.groupby(["club_id", "tidx"], sort=False).cumcount()
    win = win[win["rank"] < XI_SIZE + BENCH_SIZE]
    win = win.merge(cg.rename(columns={"idx": "tidx"})[["club_id", "tidx", "game_id", "date"]],
                    on=["club_id", "tidx"])

    # 3) T tarihindeki piyasa değeri (değerleme tarihi <= T)
    val = valuations.rename(columns={"date": "valued_at"}).sort_values("valued_at")
    win = pd.merge_asof(win.sort_values("date"), val, left_on="date", right_on="valued_at",
                        by="player_id", direction="backward", allow_exact_matches=True)
    stale = (win["date"] - win["valued_at"]).dt.days > VALUATION_MAX_AGE_DAYS
    win.loc[stale, "value_eur"] = np.nan

    # 4) T maçındaki dakika, mevki, yaş
    at_t = ap[["club_id", "game_id", "player_id", "minutes"]].rename(columns={"minutes": "minutes_at_t"})
    win = win.merge(at_t, on=["club_id", "game_id", "player_id"], how="left")
    win["minutes_at_t"] = win["minutes_at_t"].fillna(0)
    win = win.merge(players, on="player_id", how="left")
    win["age"] = (win["date"] - win["date_of_birth"]).dt.days / 365.25

    xi = win[win["rank"] < XI_SIZE].reset_index(drop=True)
    bench = win[win["rank"] >= XI_SIZE]

    # 5) Üretim ve UEFA tecrübesi: oyuncu bazında kümülatif toplamlar, pencere = fark
    stats = ap.assign(ga=ap["goals"] + ap["assists"],
                      uefa=ap["competition_id"].isin(UEFA_MAIN_COMPETITION_IDS).astype(np.int32))
    stats = stats.groupby(["player_id", "date"], as_index=False)[["minutes", "ga", "uefa"]].sum()
    stats = stats.sort_values(["player_id", "date"], kind="stable")
    for c in ("minutes", "ga", "uefa"):
        stats[f"cum_{c}"] = stats.groupby("player_id")[c].cumsum()
    stats = stats.rename(columns={"date": "stat_date"})[["player_id", "stat_date", "cum_minutes", "cum_ga", "cum_uefa"]]
    stats = stats.sort_values("stat_date")

    def cumulative_at(when: pd.Series) -> pd.DataFrame:
        q = pd.DataFrame({"rid": np.arange(len(xi)), "player_id": xi["player_id"].to_numpy(), "when": when.to_numpy()})
        m = pd.merge_asof(q.sort_values("when"), stats, left_on="when", right_on="stat_date",
                          by="player_id", direction="backward", allow_exact_matches=True)
        return m.set_index("rid").sort_index()[["cum_minutes", "cum_ga", "cum_uefa"]].fillna(0)

    now = cumulative_at(xi["date"])
    before_prod = cumulative_at(xi["date"] - pd.Timedelta(days=PRODUCTION_WINDOW_DAYS))
    before_uefa = cumulative_at(xi["date"] - pd.Timedelta(days=UEFA_EXPERIENCE_DAYS))
    xi["d_minutes"] = (now["cum_minutes"] - before_prod["cum_minutes"]).to_numpy()
    xi["d_ga"] = (now["cum_ga"] - before_prod["cum_ga"]).to_numpy()
    xi["d_uefa"] = (now["cum_uefa"] - before_uefa["cum_uefa"]).to_numpy()
    xi["v_played"] = np.where(xi["minutes_at_t"] >= CONTINUITY_MIN_MINUTES, xi["value_eur"], 0.0)

    # 6) Kulüp-maç bazında toplama
    keys = ["club_id", "tidx"]
    agg = xi.groupby(keys).agg(
        date=("date", "first"), xi_sum=("value_eur", "sum"), xi_n=("value_eur", "count"),
        xi_max=("value_eur", "max"), played_sum=("v_played", "sum"), d_minutes=("d_minutes", "sum"),
        d_ga=("d_ga", "sum"), xi_age=("age", "mean"), xi_uefa_apps=("d_uefa", "mean"))
    for position, name in (("Attack", "att"), ("Midfield", "mid"), ("Defender", "def"), ("Goalkeeper", "gk")):
        agg[f"{name}_sum"] = xi[xi["position"] == position].groupby(keys)["value_eur"].sum(min_count=1)
    bench_agg = bench.groupby(keys)["value_eur"].agg(["sum", "count"])
    agg["bench_sum"] = bench_agg["sum"]
    agg["bench_n"] = bench_agg["count"].reindex(agg.index).fillna(0)

    idx = _month_start(agg["date"]).map(value_index).to_numpy()
    ok_xi = (agg["xi_n"] >= XI_MIN_VALUED).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        def rel(col):  # log10(değer) - idx, ilk 11 değer şartıyla
            return np.where(ok_xi, np.log10(agg[col].to_numpy(float)) - idx, np.nan)
        out = pd.DataFrame({
            "xi_value": rel("xi_sum"),
            "xi_star": rel("xi_max"),
            "bench_value": np.where(agg["bench_n"].to_numpy() >= BENCH_MIN_VALUED,
                                    np.log10(agg["bench_sum"].to_numpy(float)) - idx, np.nan),
            "att_value": rel("att_sum"),
            "mid_value": rel("mid_sum"),
            "def_value": rel("def_sum"),
            "gk_value": rel("gk_sum"),
            "xi_ga90": np.where(agg["d_minutes"].to_numpy() >= PRODUCTION_MIN_MINUTES,
                                90.0 * agg["d_ga"].to_numpy(float) / agg["d_minutes"].to_numpy(float), np.nan),
            "xi_age": agg["xi_age"].to_numpy(float),
            "xi_uefa_apps": agg["xi_uefa_apps"].to_numpy(float),
            "xi_last_match_share": np.where(ok_xi, agg["played_sum"].to_numpy(float) / agg["xi_sum"].to_numpy(float), np.nan),
        }, index=agg.index)
    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    out["team"] = "tm:" + out["club_id"].astype(str)
    out["date"] = agg["date"].to_numpy()
    out["day"] = _to_days(out["date"])
    return out[["team", "date", "day", *PLAYER_STATE_COLUMNS]]


PLAYER_STATE_COLUMNS = ["xi_value", "xi_star", "bench_value", "att_value", "mid_value", "def_value",
                        "gk_value", "xi_ga90", "xi_age", "xi_uefa_apps", "xi_last_match_share"]


def build_last_match_table(matches: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    """Dinlenme günü için: takımın oynadığı her maç günü (maç kümesi + Transfermarkt kulüp maçları)."""
    tm_games = appearances[["club_id", "date"]].drop_duplicates()
    frame = pd.concat([
        matches[["home_key", "date"]].set_axis(["team", "date"], axis=1),
        matches[["away_key", "date"]].set_axis(["team", "date"], axis=1),
        pd.DataFrame({"team": "tm:" + tm_games["club_id"].astype(str), "date": tm_games["date"]}),
    ], ignore_index=True).drop_duplicates()
    frame["day"] = _to_days(frame["date"])
    frame["match_day"] = frame["day"].astype(float)
    return frame


# =============================================================================
# "Tarihten önceki son değer" indeksi — Elo, form, kadro ve oyuncu sorgularının
# ortak altyapısı. Eğitimde yüz binlerce satır, tahminde tek satır için AYNI kod.
# =============================================================================

class _AsOfIndex:
    def __init__(self, df: pd.DataFrame, key: str, day: str, values: str | list[str]):
        self.columns = [values] if isinstance(values, str) else list(values)
        self._groups = {}
        for k, g in df.sort_values(day, kind="stable").groupby(key, sort=False):
            self._groups[k] = (g[day].to_numpy(np.int64), g[self.columns].to_numpy(np.float64))

    def lookup(self, keys, when: np.ndarray, strict: bool, max_gap: int) -> np.ndarray:
        """
        Her (key, when) için day <= when (strict=True ise day < when) olan son satırın değerleri
        (satır x kolon). when - day > max_gap ise NaN. Tek kolonlu indekste 1 boyutlu döner.
        """
        keys = pd.Series(keys).reset_index(drop=True)
        when = np.asarray(when, dtype=np.int64)
        out = np.full((len(keys), len(self.columns)), np.nan)
        for k, idx in keys.groupby(keys, sort=False).groups.items():
            grp = self._groups.get(k)
            if grp is None:
                continue
            days, values = grp
            idx = np.asarray(idx)
            q = when[idx]
            pos = np.searchsorted(days, q, side="left" if strict else "right") - 1
            ok = pos >= 0
            safe = np.where(ok, pos, 0)
            ok &= (q - days[safe]) <= max_gap
            out[idx[ok]] = values[safe[ok]]
        return out[:, 0] if len(self.columns) == 1 else out


# =============================================================================
# FeatureStore — train.py ve predict.py'nin ortak giriş noktası
# =============================================================================

STORE_CACHE_PATH = dc.PROCESSED_DIR / "feature_store.joblib"


def data_signature() -> dict:
    """Ham veri dosyalarının boyut + değişiklik zamanı özeti (önbellek geçerliliği için)."""
    files = sorted(list(dc.FOOTBALL_DATA_DIR.glob("*.csv")) + list(dc.TRANSFERMARKT_DIR.glob("*.csv.gz")))
    if TEAM_ALIASES_PATH.exists():
        files.append(TEAM_ALIASES_PATH)
    return {p.name: (p.stat().st_size, int(p.stat().st_mtime)) for p in files}


class FeatureStore:
    def __init__(self, identity: TeamIdentity, elo_history: pd.DataFrame, mkt_history: pd.DataFrame,
                 team_log: pd.DataFrame, shots_log: pd.DataFrame, squad_table: pd.DataFrame,
                 player_state: pd.DataFrame, last_matches: pd.DataFrame, elo_params: dict, signature: dict):
        self.identity = identity
        self.elo_params = elo_params
        self.signature = signature
        self._elo = _AsOfIndex(elo_history, key="team", day="day", values="elo")
        self._mkt = _AsOfIndex(mkt_history, key="team", day="day", values="elo")
        self._shots = _AsOfIndex(shots_log, key="team", day="day", values=["sot_share_after", "shot_share_after"])
        self._form = _AsOfIndex(team_log, key="team", day="day", values=["form_after", "gf_after", "ga_after"])
        self._squad = _AsOfIndex(squad_table, key="team", day="day", values="squad_value")
        self._player = _AsOfIndex(player_state, key="team", day="day", values=PLAYER_STATE_COLUMNS)
        self._last_match = _AsOfIndex(last_matches, key="team", day="day", values="match_day")
        self.last_match_date = team_log["date"].max()
        self.last_player_state_date = player_state["date"].max()
        self.last_match_by_source: dict = {}

    @classmethod
    def build(cls, elo_params: dict | None = None, tune_until: pd.Timestamp | None = None) -> tuple["FeatureStore", dict]:
        """
        Diskteki verilerden store kurar. elo_params verilmezse (yalnızca eğitimde) tune_until
        tarihine kadarki veriyle ayarlanır. Dönüş: (store, bağlam).
        """
        fd = dc.load_domestic_matches()
        tm = dc.load_transfermarkt_games()
        clubs = dc.load_transfermarkt_clubs()
        identity = build_team_identity(fd, tm, clubs)
        matches = drop_same_day_duplicates(assemble_matches(fd, tm, identity))

        tuning = None
        if elo_params is None:
            if tune_until is None:
                raise ValueError("Elo parametreleri yoksa tune_until verilmeli")
            log.info("Elo parametreleri ayarlanıyor (yalnızca %s öncesi veriyle):", tune_until.date())
            elo_params, tuning = tune_elo(matches, identity.country, tune_until)
        elo = run_elo(matches, elo_params["k"], elo_params["uefa_k_mult"], identity.country)
        matches["pre_elo_home"] = elo["pre_home"]
        matches["pre_elo_away"] = elo["pre_away"]
        mkt = run_elo(matches, elo_params["mkt_k"], elo_params["uefa_k_mult"], identity.country,
                      market_weight=elo_params["mkt_w"])
        matches["pre_mkt_home"] = mkt["pre_home"]
        matches["pre_mkt_away"] = mkt["pre_away"]

        team_log = build_team_match_log(matches)
        shots_log = build_shots_log(matches)
        appearances = dc.load_transfermarkt_appearances()
        valuations = dc.load_transfermarkt_valuations()
        squad = build_squad_value_table(appearances[["player_id", "club_id", "date"]], valuations, identity.country)
        log.info("Oyuncu bazlı takım durumu hesaplanıyor...")
        value_index = build_value_index(appearances, valuations, identity.country)
        player_state = build_player_team_state(appearances, valuations, dc.load_transfermarkt_players(), value_index)
        last_matches = build_last_match_table(matches, appearances)
        store = cls(identity, elo["history"], mkt["history"], team_log, shots_log, squad, player_state,
                    last_matches, elo_params, data_signature())
        store.last_match_by_source = matches.groupby("source")["date"].max().to_dict()
        return store, {"matches": matches, "team_log": team_log, "elo_tuning": tuning,
                       "identity_report": identity.report, "appearances": appearances,
                       "valuations": valuations, "value_index": value_index}

    def save(self, path: Path = STORE_CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load_or_build(cls, elo_params: dict, path: Path = STORE_CACHE_PATH) -> "FeatureStore":
        """Önbellek aynı veri ve aynı Elo parametreleriyle üretildiyse onu kullanır, yoksa yeniden kurar."""
        if path.exists():
            try:
                store = joblib.load(path)
                if (store.signature == data_signature()
                        and {k: float(v) for k, v in store.elo_params.items()}
                        == {k: float(v) for k, v in elo_params.items()}):
                    return store
            except Exception as exc:  # noqa: BLE001 - bozuk önbellek yeniden kurulur
                log.warning("Özellik önbelleği okunamadı, yeniden kuruluyor: %s", exc)
        log.info("Özellik tabloları veriden yeniden kuruluyor (birkaç dakika sürebilir)...")
        store, _ = cls.build(elo_params=elo_params)
        store.save(path)
        return store

    def compute(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        fixtures kolonları: date, home_key, away_key, is_uefa, [neutral], [is_knockout]
        Dönüş: FEATURE_COLUMNS sırasıyla özellikler + teşhis için home_elo/away_elo.
        """
        fx = fixtures.reset_index(drop=True)
        day = _to_days(fx["date"])
        neutral = fx["neutral"].astype(float).to_numpy() if "neutral" in fx else np.zeros(len(fx))
        knockout = fx["is_knockout"].astype(float).to_numpy() if "is_knockout" in fx else np.zeros(len(fx))
        # is_uefa için sessiz varsayılan yok: yanlış değer tahmini doğrudan değiştirir
        uefa = fx["is_uefa"].astype(float).to_numpy()
        month_start = _to_days(pd.to_datetime(fx["date"]).dt.to_period("M").dt.to_timestamp())

        home_elo = self._elo.lookup(fx["home_key"], day, strict=True, max_gap=ELO_MAX_STALENESS_DAYS)
        away_elo = self._elo.lookup(fx["away_key"], day, strict=True, max_gap=ELO_MAX_STALENESS_DAYS)
        home_mkt = self._mkt.lookup(fx["home_key"], day, strict=True, max_gap=ELO_MAX_STALENESS_DAYS)
        away_mkt = self._mkt.lookup(fx["away_key"], day, strict=True, max_gap=ELO_MAX_STALENESS_DAYS)
        cols = {
            "elo_diff": home_elo + HOME_ELO_ADVANTAGE * (1.0 - neutral) - away_elo,
            "mkt_elo_diff": home_mkt + HOME_ELO_ADVANTAGE * (1.0 - neutral) - away_mkt,
            "is_knockout": knockout,
            "is_uefa": uefa,
        }
        for side in ("home", "away"):
            key = fx[f"{side}_key"]
            shots = self._shots.lookup(key, day, strict=True, max_gap=FORM_MAX_GAP_DAYS)
            cols[f"{side}_sot_share"] = shots[:, 0]
            cols[f"{side}_shot_share"] = shots[:, 1]
            form = self._form.lookup(key, day, strict=True, max_gap=FORM_MAX_GAP_DAYS)
            cols[f"{side}_form"] = form[:, 0]
            cols[f"{side}_goals_for"] = form[:, 1]
            cols[f"{side}_goals_against"] = form[:, 2]
            # ay başı anlık görüntüsü: tam olarak D'nin ayına ait olmalı (gecikme <= 0 gün, ay içi)
            cols[f"{side}_squad_value"] = self._squad.lookup(key, month_start, strict=False, max_gap=0)
            player = self._player.lookup(key, day, strict=True, max_gap=PLAYER_STATE_MAX_GAP_DAYS)
            for j, name in enumerate(PLAYER_STATE_COLUMNS):
                cols[f"{side}_{name}"] = player[:, j]
            last_day = self._last_match.lookup(key, day, strict=True, max_gap=np.iinfo(np.int32).max)
            cols[f"{side}_rest_days"] = np.minimum(day - last_day, REST_DAYS_CAP)
        cols["xi_value_diff"] = cols["home_xi_value"] - cols["away_xi_value"]
        out = pd.DataFrame(cols)[FEATURE_COLUMNS].astype("float64")
        out["home_elo"] = home_elo
        out["away_elo"] = away_elo
        out["home_mkt_elo"] = home_mkt
        out["away_mkt_elo"] = away_mkt
        return out


# =============================================================================
# Sızıntı / tutarlılık kontrolleri (eğitim bunları geçmezse durur)
# =============================================================================

def assert_form_has_no_leakage(matches: pd.DataFrame, features: pd.DataFrame) -> None:
    """Form ve gol formu, açık (bağımsız yazılmış) shift(1).rolling(...) hesabıyla birebir aynı olmalı."""
    log_df = build_team_match_log(matches)
    checks = (("points", FORM_WINDOW, FORM_MIN_MATCHES, "form"),
              ("goals_for", GOALS_FORM_WINDOW, GOALS_FORM_MIN_MATCHES, "goals_for"),
              ("goals_against", GOALS_FORM_WINDOW, GOALS_FORM_MIN_MATCHES, "goals_against"))
    for col, window, min_periods, feature in checks:
        explicit = log_df.groupby(["team", "segment"])[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        ref = pd.Series(explicit.to_numpy(), index=pd.MultiIndex.from_frame(log_df[["team", "date"]]))
        for side in ("home", "away"):
            idx = pd.MultiIndex.from_arrays([matches[f"{side}_key"], matches["date"]])
            same = np.isclose(ref.reindex(idx).to_numpy(), features[f"{side}_{feature}"].to_numpy(), equal_nan=True)
            if not same.all():
                raise AssertionError(f"{side}_{feature} özelliğinde sızıntı/tutarsızlık: "
                                     f"{int((~same).sum())} satır uyuşmuyor.")

    # İsabetli şut payı: bağımsız shift(1).rolling(...).sum() oranı; şut verisi olmayan maçlarda
    # özellik, önceki şut verili maçlardan gelir -> karşılaştırma yalnızca şut verili maçlarda yapılır.
    shots = build_shots_log(matches)
    grp = shots.groupby(["team", "segment"])
    num = grp["sot_for"].transform(lambda s: s.shift(1).rolling(SHOTS_WINDOW, min_periods=SHOTS_MIN_MATCHES).sum())
    den = grp["sot_total"].transform(lambda s: s.shift(1).rolling(SHOTS_WINDOW, min_periods=SHOTS_MIN_MATCHES).sum())
    ref = pd.Series((num / den).where(den > 0).to_numpy(), index=pd.MultiIndex.from_frame(shots[["team", "date"]]))
    has_shots = matches[["home_sot", "away_sot", "home_shots", "away_shots"]].notna().all(axis=1).to_numpy()
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([matches[f"{side}_key"], matches["date"]])
        expected = ref.reindex(idx).to_numpy()
        same = np.isclose(expected, features[f"{side}_sot_share"].to_numpy(), equal_nan=True)
        if not same[has_shots].all():
            raise AssertionError(f"{side}_sot_share özelliğinde sızıntı/tutarsızlık: "
                                 f"{int((~same[has_shots]).sum())} satır uyuşmuyor.")


def assert_player_state_no_leakage(matches: pd.DataFrame, features: pd.DataFrame, appearances: pd.DataFrame,
                                   valuations: pd.DataFrame, value_index: pd.Series,
                                   n_samples: int = 300, seed: int = 0) -> None:
    """
    Bağımsız yeniden hesaplama: rastgele maçlarda home_xi_value ve home_xi_ga90, YALNIZCA maç tarihinden
    önceki appearances/valuations satırları filtrelenerek satır satır hesaplanır ve toplu (vektörize)
    hesaplamayla karşılaştırılır. Maçın kendisi ya da sonrası kullanılmışsa değerler tutmaz.
    """
    tm_rows = matches.index[matches["home_key"].str.startswith("tm:")]
    rng = np.random.default_rng(seed)
    sample = rng.choice(tm_rows, size=min(n_samples, len(tm_rows)), replace=False)
    by_club = {k: g for k, g in appearances.groupby("club_id")}
    val_by_player = {k: g.sort_values("date") for k, g in valuations.groupby("player_id")}
    mismatches = 0
    for i in sample:
        club = int(matches.at[i, "home_key"][3:])
        when = matches.at[i, "date"]
        ap = by_club.get(club)
        exp_value = exp_ga90 = np.nan
        if ap is not None:
            past = ap[ap["date"] < when]
            games = past[["game_id", "date"]].drop_duplicates().sort_values(["date", "game_id"])
            if len(games) and (when - games["date"].iloc[-1]).days <= PLAYER_STATE_MAX_GAP_DAYS:
                t = games["date"].iloc[-1]
                recent = past[past["game_id"].isin(games["game_id"].iloc[-PLAYER_WINDOW_MATCHES:])]
                mins = recent.groupby("player_id")["minutes"].sum().reset_index()
                xi = mins.sort_values(["minutes", "player_id"], ascending=[False, True]).head(XI_SIZE)["player_id"]
                values = []
                for p in xi:
                    v = val_by_player.get(p)
                    v = v[v["date"] <= t] if v is not None else None
                    if v is not None and len(v) and (t - v["date"].iloc[-1]).days <= VALUATION_MAX_AGE_DAYS:
                        values.append(v["value_eur"].iloc[-1])
                if len(values) >= XI_MIN_VALUED:
                    exp_value = np.log10(sum(values)) - value_index.get(t.to_period("M").to_timestamp(), np.nan)
                # üretim: oyuncunun TÜM kulüplerdeki maçları (t - pencere, t]
                window = appearances[appearances["player_id"].isin(xi)
                                     & (appearances["date"] <= t)
                                     & (appearances["date"] > t - pd.Timedelta(days=PRODUCTION_WINDOW_DAYS))]
                if window["minutes"].sum() >= PRODUCTION_MIN_MINUTES:
                    exp_ga90 = 90.0 * (window["goals"] + window["assists"]).sum() / window["minutes"].sum()
        got_value, got_ga90 = features.at[i, "home_xi_value"], features.at[i, "home_xi_ga90"]
        if not (np.isclose(exp_value, got_value, equal_nan=True) and np.isclose(exp_ga90, got_ga90, equal_nan=True)):
            mismatches += 1
            if mismatches <= 3:
                log.error("Oyuncu durumu uyuşmazlığı: %s %s beklenen=(%s, %s) bulunan=(%s, %s)",
                          matches.at[i, "home_team"], when.date(), exp_value, exp_ga90, got_value, got_ga90)
    if mismatches:
        raise AssertionError(f"Oyuncu bazlı özelliklerde {mismatches}/{len(sample)} örnek bağımsız hesapla uyuşmuyor.")


def assert_elo_consistency(matches: pd.DataFrame, features: pd.DataFrame) -> None:
    """
    Tarih sorgusuyla (predict yolu) bulunan Elo, motorun maç öncesi rating'iyle aynı olmalı.
    Bu, hem eğitim/tahmin tutarlılığını hem de "maç sonucu kendi özelliğine sızmadı" şartını doğrular.
    """
    for engine, feature_col in (("elo", "elo"), ("mkt", "mkt_elo")):
        for side in ("home", "away"):
            got = features[f"{side}_{feature_col}"].to_numpy()
            expected = matches[f"pre_{engine}_{side}"].to_numpy()
            known = ~np.isnan(got)
            bad = ~np.isclose(got[known], expected[known])
            if bad.any():
                raise AssertionError(f"{engine} tutarsızlığı ({side}): {int(bad.sum())} satır motorla uyuşmuyor.")
