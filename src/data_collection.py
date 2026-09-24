"""
Veri toplama katmanı: ham verileri indirir, `data/` altına önbelleğe alır ve
standart şemalı pandas DataFrame'leri olarak yükler.

Kaynaklar
---------
1. football-data.co.uk -> 21 Avrupa liginin iç lig maçları (skor, hakem, kart, faul, oranlar)
2. Transfermarkt veri seti (dcaribou/transfermarkt-datasets, CC0) ->
     - UEFA Şampiyonlar Ligi / Avrupa Ligi / Konferans Ligi maçları + eleme turları
     - football-data'da olmayan ligler (Ukrayna, Hırvatistan, Çekya, Sırbistan)
     - oyuncu maç kadroları (appearances) ve piyasa değeri geçmişi (player_valuations)
3. Sakatlık / kart cezası -> henüz yok, placeholder

Neden ClubElo değil: api.clubelo.com rating endpoint'leri sürekli 502 döndürdü.
Ligler arası Elo bu yüzden features.py içinde, iç lig + UEFA maçlarından hesaplanır.

Kullanım (proje kökünden):
    python -m src.data_collection                 # hepsini indir / güncelle
    python -m src.data_collection --only football-data
    python -m src.data_collection --only transfermarkt
    python -m src.data_collection --check         # sadece neyin mevcut olduğunu raporla

Kural: Bir kaynaktan veri alınamazsa kod SAHTE veri üretmez; açık hata / uyarı verir.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

FOOTBALL_DATA_DIR = DATA_DIR / "football_data"
TRANSFERMARKT_DIR = DATA_DIR / "transfermarkt"
REPORTS_DIR = DATA_DIR / "reports"
PROCESSED_DIR = DATA_DIR / "processed"

HTTP_HEADERS = {"User-Agent": "futbol-ml-modeli/1.0 (egitim amacli veri indirme)"}
HTTP_TIMEOUT = 60

# Veri başlangıcı: football-data'da ortalama oranlar 2008-09'dan itibaren tüm liglerde var.
# 2008-2014 arası Elo / piyasa Elo'sunun "ısınma" dönemidir (UEFA maçları Transfermarkt'ta 2012
# yazında başlar); eğitim satırları TRAIN_FIRST_SEASON'dan başlar (oyuncu verisi 2012+).
DEFAULT_FIRST_SEASON_START = 2008
TRAIN_FIRST_SEASON_START = 2014

# Ülke kodları (tüm kaynaklarda ortak)
TM_COUNTRY_CODES = {
    "England": "ENG", "Scotland": "SCO", "Germany": "GER", "Italy": "ITA", "Spain": "ESP",
    "France": "FRA", "Netherlands": "NED", "Belgium": "BEL", "Portugal": "POR",
    "Türkiye": "TUR", "Turkey": "TUR", "Greece": "GRE", "Russia": "RUS", "Ukraine": "UKR",
    "Denmark": "DEN", "Austria": "AUT", "Switzerland": "SUI", "Poland": "POL",
    "Romania": "ROU", "Croatia": "CRO", "Czech Republic": "CZE", "Serbia": "SRB",
    "Norway": "NOR", "Sweden": "SWE",
}

# --- football-data.co.uk lig tanımları ---------------------------------------
# "Ana" ligler: sezon başına bir dosya -> mmz4281/{SSEE}/{kod}.csv
MAIN_LEAGUES = {
    "E0": "ENG",   # Premier League
    "SC0": "SCO",  # Scottish Premiership
    "D1": "GER",   # Bundesliga
    "I1": "ITA",   # Serie A
    "SP1": "ESP",  # La Liga
    "F1": "FRA",   # Ligue 1
    "N1": "NED",   # Eredivisie
    "B1": "BEL",   # Belçika Pro League
    "P1": "POR",   # Primeira Liga
    "T1": "TUR",   # Süper Lig
    "G1": "GRE",   # Yunanistan Super League
    # 2. ligler: yalnızca geçmiş (Elo, form, piyasa Elo'su) için; eğitim satırı olarak kullanılmaz.
    # Terfi eden takımların ratingi "yeni kulüp" varsayımı yerine gerçek maçlardan gelir.
    "E1": "ENG",   # Championship
    "SC1": "SCO",  # Scottish Championship
    "D2": "GER",   # 2. Bundesliga
    "I2": "ITA",   # Serie B
    "SP2": "ESP",  # Segunda División
    "F2": "FRA",   # Ligue 2
}
SECOND_TIER_LEAGUES = {"E1", "SC1", "D2", "I2", "SP2", "F2"}

# "Extra" ligler: tüm sezonlar tek dosyada -> new/{kod}.csv (hakem/kart verisi yok)
EXTRA_LEAGUES = {
    "AUT": "AUT",  # Avusturya
    "DNK": "DEN",  # Danimarka
    "FIN": "FIN",  # Finlandiya
    "IRL": "IRL",  # İrlanda
    "NOR": "NOR",  # Norveç
    "POL": "POL",  # Polonya
    "ROU": "ROU",  # Romanya
    "RUS": "RUS",  # Rusya
    "SWE": "SWE",  # İsveç
    "SWZ": "SUI",  # İsviçre
}
FOOTBALL_DATA_COUNTRIES = set(MAIN_LEAGUES.values()) | set(EXTRA_LEAGUES.values())

# Standart maç şeması (tüm kaynaklar bu kolonlara dönüştürülür)
MATCH_COLUMNS = [
    "date", "season_start", "source", "competition", "tier", "is_uefa", "country",
    "home_team", "away_team", "home_tm_id", "away_tm_id",
    "home_goals", "away_goals", "result", "is_knockout", "neutral",
    "referee", "home_fouls", "away_fouls", "home_yellow", "away_yellow",
    "home_red", "away_red", "home_shots", "away_shots", "home_sot", "away_sot",
    "odds_home", "odds_draw", "odds_away", "odds_type",
]

# Oran önceliği: kapanış oranları maç öncesi son piyasa görüşüdür ve en isabetlisidir.
# Pinnacle kapanış (PSC) > ortalama kapanış (AvgC) > Bet365 kapanış (B365C) > maç öncesi
# ortalama (Avg / BetBrain BbAv) > Bet365 > Pinnacle maç öncesi (PS)
ODDS_PRIORITY = ("PSC", "AvgC", "B365C", "Avg", "BbAv", "B365", "PS")


# =============================================================================
# Yardımcılar
# =============================================================================

def season_start_year(ts) -> int:
    """Avrupa sezonu Temmuz'da başlar: 2026-09-14 -> 2026, 2027-03-01 -> 2026."""
    return ts.year if ts.month >= 7 else ts.year - 1


def _season_code(start_year: int) -> str:
    """2014 -> '1415' (football-data URL formatı)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _http_get(url: str, retries: int = 3, backoff: float = 2.0) -> requests.Response:
    """Basit tekrar denemeli GET. 4xx hatalarında tekrar denemez (anlamsız)."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if 400 <= resp.status_code < 500:
                resp.raise_for_status()
            last_exc = requests.HTTPError(f"HTTP {resp.status_code} - {url}")
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < retries:
            time.sleep(backoff * attempt)
    raise ConnectionError(f"İndirilemedi ({retries} deneme): {url} -> {last_exc}")


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """football-data dosyaları eski sezonlarda latin-1, yenilerde UTF-8 BOM'lu."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    # Bazı eski dosyalarda satır sonunda fazladan virgül/bozuk satır var; bunlar
    # atlanır. Tamamen boş satırlar (",,,,") aşağıda dropna ile temizlenir.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)


def _result_from_goals(df: pd.DataFrame) -> pd.Series:
    res = pd.Series("D", index=df.index)
    res[df["home_goals"] > df["away_goals"]] = "H"
    res[df["home_goals"] < df["away_goals"]] = "A"
    return res


# =============================================================================
# 1) football-data.co.uk
# =============================================================================

def download_football_data(first_season: int = DEFAULT_FIRST_SEASON_START,
                           last_season: int | None = None,
                           force: bool = False) -> list[Path]:
    """
    Ana ligleri sezon sezon, extra ligleri tek dosya olarak indirir.
    Tamamlanmış sezon dosyaları varsa tekrar indirilmez; devam eden sezon
    (ve extra lig dosyaları, çünkü içlerinde güncel sezon var) her çağrıda yenilenir.
    """
    last_season = last_season or season_start_year(date.today())
    current = season_start_year(date.today())
    FOOTBALL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    failures: list[str] = []

    for start in range(first_season, last_season + 1):
        code = _season_code(start)
        for div in MAIN_LEAGUES:
            path = FOOTBALL_DATA_DIR / f"{code}_{div}.csv"
            if path.exists() and not force and start < current:
                saved.append(path)
                continue
            url = f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
            try:
                path.write_bytes(_http_get(url).content)
                saved.append(path)
                log.info("indirildi: %s", url)
            except Exception as exc:  # noqa: BLE001 - tek dosya hatası tüm indirmeyi durdurmasın
                failures.append(f"{url} -> {exc}")
            time.sleep(0.3)  # sunucuya nazik davran

    for code in EXTRA_LEAGUES:
        path = FOOTBALL_DATA_DIR / f"extra_{code}.csv"
        url = f"https://www.football-data.co.uk/new/{code}.csv"
        try:
            path.write_bytes(_http_get(url).content)
            saved.append(path)
            log.info("indirildi: %s", url)
        except Exception as exc:  # noqa: BLE001
            if path.exists():
                saved.append(path)
            failures.append(f"{url} -> {exc}")
        time.sleep(0.3)

    if failures:
        log.warning("football-data: %d dosya indirilemedi:\n  %s",
                    len(failures), "\n  ".join(failures))
    return saved


def _best_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her maç için ODDS_PRIORITY sırasındaki ilk GEÇERLİ (üçü de > 1) oran üçlüsü.
    Dosya içinde bile kolon dolulukları sezona göre değiştiği için satır bazında seçilir.
    """
    out = pd.DataFrame({"odds_home": np.nan, "odds_draw": np.nan, "odds_away": np.nan,
                        "odds_type": pd.Series([None] * len(df), dtype=object)}, index=df.index)
    for prefix in ODDS_PRIORITY:
        cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
        if not all(c in df.columns for c in cols):
            continue
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        valid = (vals > 1).all(axis=1) & out["odds_home"].isna()
        out.loc[valid, ["odds_home", "odds_draw", "odds_away"]] = vals[valid].to_numpy()
        out.loc[valid, "odds_type"] = prefix
    return out


def _standardize_main(df: pd.DataFrame, div: str) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    col = lambda name: df[name] if name in df.columns else pd.NA  # noqa: E731
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce"),
        "country": MAIN_LEAGUES[div],
        "competition": f"league:{div}",
        "tier": 2 if div in SECOND_TIER_LEAGUES else 1,
        "home_team": df["HomeTeam"].astype(str).str.strip(),
        "away_team": df["AwayTeam"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
        "referee": col("Referee"),
        "home_fouls": col("HF"), "away_fouls": col("AF"),
        "home_yellow": col("HY"), "away_yellow": col("AY"),
        "home_red": col("HR"), "away_red": col("AR"),
        "home_shots": col("HS"), "away_shots": col("AS"),
        "home_sot": col("HST"), "away_sot": col("AST"),
    })
    return pd.concat([out, _best_odds(df)], axis=1)


def _standardize_extra(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df.dropna(subset=["Home", "Away", "HG", "AG"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce"),
        "country": EXTRA_LEAGUES[code],
        "competition": f"league:{code}",
        "tier": 1,
        "home_team": df["Home"].astype(str).str.strip(),
        "away_team": df["Away"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(df["HG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["AG"], errors="coerce"),
    })
    return pd.concat([out, _best_odds(df)], axis=1)


def load_domestic_matches(first_season: int = DEFAULT_FIRST_SEASON_START) -> pd.DataFrame:
    """
    İndirilmiş tüm football-data dosyalarını tek bir standart DataFrame'e çevirir.
    home_tm_id/away_tm_id burada boştur; takım kimliği features.py'de eşlenir.
    """
    if not FOOTBALL_DATA_DIR.exists() or not any(FOOTBALL_DATA_DIR.glob("*.csv")):
        raise FileNotFoundError(
            f"football-data dosyası yok: {FOOTBALL_DATA_DIR}\n"
            "Önce `python -m src.data_collection --only football-data` çalıştırın."
        )
    frames = []
    for path in sorted(FOOTBALL_DATA_DIR.glob("*.csv")):
        raw = _read_csv_bytes(path.read_bytes())
        if path.name.startswith("extra_"):
            code = path.stem.removeprefix("extra_")
            if code in EXTRA_LEAGUES and "Home" in raw.columns:
                frames.append(_standardize_extra(raw, code))
        else:
            div = path.stem.split("_", 1)[1]
            if div in MAIN_LEAGUES and "HomeTeam" in raw.columns:
                frames.append(_standardize_main(raw, div))
            else:
                log.warning("Tanınmayan dosya formatı atlandı: %s", path.name)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["season_start"] = df["date"].map(season_start_year).astype(int)
    df = df[df["season_start"] >= first_season].copy()
    df["result"] = _result_from_goals(df)
    df["source"] = "football-data"
    df["is_uefa"] = False
    df["is_knockout"] = 0
    df["neutral"] = 0
    df["home_tm_id"] = pd.NA
    df["away_tm_id"] = pd.NA
    for c in ["home_fouls", "away_fouls", "home_yellow", "away_yellow", "home_red", "away_red",
              "home_shots", "away_shots", "home_sot", "away_sot", "odds_home", "odds_draw", "odds_away"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df["tier"] = df["tier"].astype(int)
    # Aynı maç iki dosyada varsa (ör. yeniden indirme) tekilleştir
    df = df.drop_duplicates(subset=["date", "country", "home_team", "away_team"])
    return df.sort_values(["date", "country", "home_team"]).reset_index(drop=True)[MATCH_COLUMNS]


# =============================================================================
# 2) Transfermarkt veri seti (dcaribou/transfermarkt-datasets, CC0)
# =============================================================================

TRANSFERMARKT_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TRANSFERMARKT_TABLES = ["games", "competitions", "clubs", "appearances", "player_valuations", "players"]

UEFA_MAIN_COMPETITIONS = {"CL", "EL", "UCOL"}          # grup/lig aşaması + eleme turları
UEFA_QUALIFYING_COMPETITIONS = {"CLQ", "ELQ", "ECLQ"}  # grup öncesi ön eleme turları


def download_transfermarkt(force: bool = False) -> list[Path]:
    """
    Veri seti tablolarını (csv.gz) indirir. Sunucudaki dosya diskteki ile aynı boyuttaysa
    ve `force` verilmediyse tekrar indirmez.
    """
    TRANSFERMARKT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for table in TRANSFERMARKT_TABLES:
        url = f"{TRANSFERMARKT_BASE_URL}/{table}.csv.gz"
        path = TRANSFERMARKT_DIR / f"{table}.csv.gz"
        try:
            head = requests.head(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
            remote_size = int(head.headers.get("Content-Length", -1))
            remote_modified = head.headers.get("Last-Modified", "?")
        except requests.RequestException as exc:
            if path.exists():
                log.warning("Transfermarkt sunucusuna erişilemedi, mevcut dosya kullanılacak: %s (%s)", table, exc)
                saved.append(path)
                continue
            raise ConnectionError(f"Transfermarkt veri setine erişilemedi: {url} -> {exc}") from exc
        if path.exists() and not force and path.stat().st_size == remote_size:
            saved.append(path)
            continue
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=600)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        log.info("indirildi: %s (%.1f MB, sunucu tarihi %s)", table, len(resp.content) / 1e6, remote_modified)
        saved.append(path)
    return saved


def _require_tm_table(table: str) -> Path:
    path = TRANSFERMARKT_DIR / f"{table}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Transfermarkt tablosu yok: {path}\n"
            "`python -m src.data_collection --only transfermarkt` çalıştırın."
        )
    return path


def _classify_uefa_round(competition_id: str, round_name: str, season: int,
                         when: pd.Timestamp) -> tuple[int, int]:
    """
    (is_knockout, neutral) döner.
    - Ön eleme turları (CLQ/ELQ/ECLQ): grup öncesi -> is_knockout = 0
    - Ana turnuvada "Group ..." / "Group Stage" (2024-25'ten itibaren lig aşaması) -> 0
    - Ana turnuvada geri kalan her tur (play-off/intermediate, son 16, çeyrek, yarı, final) -> 1
    - Tarafsız saha: finaller ve 2019-20'de Ağustos 2020'deki tek maçlık COVID turları
      (Lizbon / Almanya; turda "leg" geçmeyen maçlar)
    """
    if competition_id in UEFA_QUALIFYING_COMPETITIONS:
        return 0, 0
    r = str(round_name).strip().lower()
    if r.startswith("group"):
        return 0, 0
    neutral = int(r == "final" or (season == 2019 and when >= pd.Timestamp(2020, 8, 1) and "leg" not in r))
    return 1, neutral


def load_transfermarkt_clubs() -> pd.DataFrame:
    """club_id, name, country (kulübün Transfermarkt'taki iç ligi üzerinden; bilinmiyorsa boş)."""
    clubs = pd.read_csv(_require_tm_table("clubs"))
    comps = pd.read_csv(_require_tm_table("competitions"))
    country = comps.set_index("competition_id")["country_name"].map(TM_COUNTRY_CODES)
    clubs["country"] = clubs["domestic_competition_id"].map(country)
    return clubs[["club_id", "name", "country"]]


def load_transfermarkt_games(first_season: int = DEFAULT_FIRST_SEASON_START) -> pd.DataFrame:
    """
    Transfermarkt maçlarından UEFA (ana + ön eleme) ve Avrupa iç lig maçlarını standart şemaya çevirir.
    İç kupalar ve süper kupalar alınmaz (alt lig takımları + tek maç formatı).
    """
    games = pd.read_csv(_require_tm_table("games"), low_memory=False)
    comps = pd.read_csv(_require_tm_table("competitions"))
    comp_country = comps.set_index("competition_id")["country_name"].map(TM_COUNTRY_CODES)
    domestic_ids = set(comps.loc[comps["type"] == "domestic_league", "competition_id"])

    uefa_ids = UEFA_MAIN_COMPETITIONS | UEFA_QUALIFYING_COMPETITIONS
    g = games[games["competition_id"].isin(uefa_ids | domestic_ids)].copy()
    g = g.dropna(subset=["date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"])
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"])
    g["country"] = g["competition_id"].map(comp_country)
    # Avrupa dışı ligler (ARG1, BRA1, MLS1, ...) ülke kodu olmadığı için düşer
    g = g[g["competition_id"].isin(uefa_ids) | g["country"].notna()]

    is_uefa = g["competition_id"].isin(uefa_ids)
    ko_neutral = [
        _classify_uefa_round(c, r, s, d) if u else (0, 0)
        for c, r, s, d, u in zip(g["competition_id"], g["round"], g["season"], g["date"], is_uefa)
    ]
    out = pd.DataFrame({
        "date": g["date"],
        "source": "transfermarkt",
        "competition": [("uefa:" if u else "league:") + c for c, u in zip(g["competition_id"], is_uefa)],
        "is_uefa": is_uefa.to_numpy(),
        "country": g["country"].where(~is_uefa, None),
        "home_team": g["home_club_name"].astype(str),
        "away_team": g["away_club_name"].astype(str),
        "home_tm_id": g["home_club_id"].astype(int),
        "away_tm_id": g["away_club_id"].astype(int),
        "home_goals": g["home_club_goals"].astype(int),
        "away_goals": g["away_club_goals"].astype(int),
        "is_knockout": [k for k, _ in ko_neutral],
        "neutral": [n for _, n in ko_neutral],
        "referee": g["referee"],
        "tier": 1,
    })
    out["season_start"] = out["date"].map(season_start_year).astype(int)
    out = out[out["season_start"] >= first_season].copy()
    out["result"] = _result_from_goals(out)
    for c in MATCH_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out.sort_values("date").reset_index(drop=True)[MATCH_COLUMNS]


def load_transfermarkt_appearances() -> pd.DataFrame:
    """
    Oyuncunun forma giydiği her maç: game_id, player_id, club_id (oyuncunun O maçtaki kulübü), date,
    competition_id, minutes, goals, assists. Transfermarkt veri setindeki tüm turnuvaları kapsar
    (iç ligler, iç kupalar, UEFA).
    """
    df = pd.read_csv(_require_tm_table("appearances"),
                     usecols=["game_id", "player_id", "player_club_id", "date", "competition_id",
                              "minutes_played", "goals", "assists"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["game_id", "player_id", "player_club_id", "date"])
    df = df.rename(columns={"player_club_id": "club_id", "minutes_played": "minutes"})
    for c in ("minutes", "goals", "assists"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int32)
    return df.astype({"game_id": np.int64, "player_id": np.int64, "club_id": np.int64})


def load_transfermarkt_players() -> pd.DataFrame:
    """
    player_id, position (Goalkeeper / Defender / Midfield / Attack / Missing), date_of_birth.
    Not: international_caps, market_value_in_eur gibi kolonlar veri setinin indirildiği ANDAKİ
    değerlerdir; geçmiş maçlar için sızıntı yaratacağından bilerek alınmaz.
    """
    df = pd.read_csv(_require_tm_table("players"), usecols=["player_id", "position", "date_of_birth"])
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    return df.astype({"player_id": np.int64})


def load_transfermarkt_valuations() -> pd.DataFrame:
    """player_id, date, value_eur (Transfermarkt piyasa değeri tahmini; kitle kaynaklı)."""
    df = pd.read_csv(_require_tm_table("player_valuations"),
                     usecols=["player_id", "date", "market_value_in_eur"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna().rename(columns={"market_value_in_eur": "value_eur"})
    return df[df["value_eur"] > 0].astype({"player_id": int})


# =============================================================================
# 3) Sakatlık / kart cezası (placeholder)
# =============================================================================

def load_injuries_and_suspensions() -> pd.DataFrame | None:
    """
    PLACEHOLDER — sonraki aşama (aday kaynak: API-Football /injuries, API anahtarı gerekir).
    Hedef şema: date, club, player, type ('injury' | 'suspension'), expected_return.
    Şimdilik hiçbir özellik bunu kullanmaz.
    """
    return None


# =============================================================================
# Durum raporu ve CLI
# =============================================================================

def data_status() -> dict:
    """Hangi verinin mevcut olduğunu özetler (hiçbir şey indirmez)."""
    fd_files = list(FOOTBALL_DATA_DIR.glob("*.csv")) if FOOTBALL_DATA_DIR.exists() else []
    tm = {t: (TRANSFERMARKT_DIR / f"{t}.csv.gz").exists() for t in TRANSFERMARKT_TABLES}
    return {
        "football_data_files": len(fd_files),
        "transfermarkt_tables": f"{sum(tm.values())}/{len(tm)}"
                                + ("" if all(tm.values()) else f" eksik: {[t for t, ok in tm.items() if not ok]}"),
        "injuries_source": "yok (placeholder)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ham veri indirme")
    parser.add_argument("--only", choices=["football-data", "transfermarkt"], default=None)
    parser.add_argument("--first-season", type=int, default=DEFAULT_FIRST_SEASON_START,
                        help="İlk sezonun başlangıç yılı (varsayılan 2012 = 2012-13)")
    parser.add_argument("--force", action="store_true", help="Önbelleği yok say")
    parser.add_argument("--check", action="store_true", help="Sadece durum raporu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.check:
        if args.only in (None, "football-data"):
            files = download_football_data(args.first_season, force=args.force)
            log.info("football-data: %d dosya hazır", len(files))
        if args.only in (None, "transfermarkt"):
            files = download_transfermarkt(force=args.force)
            log.info("transfermarkt: %d tablo hazır", len(files))
    for k, v in data_status().items():
        log.info("%-24s %s", k, v)


if __name__ == "__main__":
    main()
