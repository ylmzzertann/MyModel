# feature_schema.md

> **Bu dosya `src/train.py` tarafından otomatik üretilir — elle düzenlemeyin.**
> `models/mac_modeli.pkl` ile birlikte taşınmalıdır. `predict.py`, en alttaki JSON bloğunu
> okuyup modelin beklediği özelliklerle ve topluluk parametreleriyle birebir karşılaştırır.

- **Eğitim zamanı (UTC):** 2026-09-21 06:47:02
- **Sözleşme hash'i:** `0589222bbcf9097f483ac2d818772ff33373ce8c81e9ef3949df6684bff524cd`
- **Model girdisi:** tek satırlık `pandas.DataFrame`, kolonlar aşağıdaki sırada, hepsi `float64`
- **Model çıktısı:** `[home_win, draw, away_win]` olasılıkları + (opsiyonel) skor matrisi

## Model yapısı (topluluk)

```text
p_xgb     = components['xgb'].predict_proba(X)                  # XGBoost sınıflandırıcı
p_logreg  = components['logreg'].predict_proba(X)               # Lojistik Regresyon pipeline (imputer+scaler)
λ_ev      = components['poisson_home'].predict(X)               # XGBoost Poisson regresyonu (ev golü)
λ_dep     = components['poisson_away'].predict(X)               # XGBoost Poisson regresyonu (deplasman golü)
M[i,j]    = Poisson(i; λ_ev) * Poisson(j; λ_dep), i,j = 0..10
            Dixon-Coles: M[0,0]*=1-λ_ev*λ_dep*ρ; M[0,1]*=1+λ_ev*ρ; M[1,0]*=1+λ_dep*ρ; M[1,1]*=1-ρ
            negatifler 0'a kırpılır, M toplamı 1'e normalize edilir
p_poisson = [Σ_{i>j} M, Σ_{i=j} M, Σ_{i<j} M]
p         = w_xgb*p_xgb + w_logreg*p_logreg + w_poisson*p_poisson ; p /= Σp
p_final   = softmax(log(max(p, 1e-6)) / T)
skor dağılımı (opsiyonel): M'nin ev/beraberlik/deplasman bölgeleri p_final'a eşit toplamlara ölçeklenir
```

- Ağırlıklar: `{"xgb": 0.45, "logreg": 0.25, "poisson": 0.3}`, Dixon-Coles ρ = `-0.03044`, sıcaklık T = `0.91929`
- Bileşen hiperparametreleri: `{"xgb": {"learning_rate": 0.03, "max_depth": 5, "min_child_weight": 100, "colsample_bytree": 0.8, "reg_lambda": 2.0, "half_life": 4.0, "n_estimators": 232}, "logreg": {"C": 1.0, "half_life": 4.0}, "poisson": {"learning_rate": 0.03, "max_depth": 4, "min_child_weight": 50, "colsample_bytree": 0.8, "reg_lambda": 2.0, "half_life": 4.0, "n_estimators": 328}}`
- Formülün referans uygulaması: `src/predict.py` (`component_outputs`, `combine`, `consistent_score_matrix`).

## Veri kaynakları ve aralık

- football-data.co.uk: 21 Avrupa 1. ligi + 6 adet 2. lig (maç sonuçları, kapanış oranları, şutlar)
- Transfermarkt veri seti (dcaribou/transfermarkt-datasets, CC0): UEFA CL/EL/UECL + ön eleme turları, UKR/CRO/CZE/SRB ligleri, oyuncu maç kadroları, piyasa değerleri, oyuncu mevkileri
- Kaynakların son maç tarihleri: {'football-data': '2026-09-17', 'transfermarkt': '2026-05-24'}
- Elo ısınma dönemi: 2008-07-01 - 2014-07-01
- Eğitim satırları: 1. lig + UEFA maçları (2. lig maçları yalnızca geçmiş özellikleri besler)

| Bölüm | Başlangıç | Bitiş | Maç sayısı | UEFA maçı |
|---|---|---|---|---|
| development | 2014-07-01 | 2025-06-30 | 73281 | 7736 |
| test | 2025-07-01 | 2026-09-17 | 8519 | 927 |

Çapraz doğrulama sezonları: ['2021-07-01 - 2022-07-01', '2022-07-01 - 2023-07-01', '2023-07-01 - 2024-07-01', '2024-07-01 - 2025-07-01']. Kaydedilen model şu veriyle eğitildi: **geliştirme + test (--refit-all; test metrikleri bir önceki fit'e ait)**.

## Özellikler (sıra önemlidir)

| # | Ad | Grup | Tip | Kaynak | Eğitimde NaN oranı |
|---|---|---|---|---|---|
| 1 | `elo_diff` | base | float64 | Kendi ligler arası Elo (football-data + Transfermarkt maç sonuçları) | 0.0% |
| 2 | `home_form` | form | float64 | football-data.co.uk + Transfermarkt maç sonuçları | 2.4% |
| 3 | `away_form` | form | float64 | football-data.co.uk + Transfermarkt maç sonuçları | 2.5% |
| 4 | `is_knockout` | base | float64 | Transfermarkt games.round | 0.0% |
| 5 | `is_uefa` | base | float64 | maç kaynağı (Transfermarkt competition_id) | 0.0% |
| 6 | `home_att_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 33.6% |
| 7 | `away_att_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 33.6% |
| 8 | `home_mid_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 33.2% |
| 9 | `away_mid_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 33.2% |
| 10 | `home_def_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 33.1% |
| 11 | `away_def_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 33.1% |
| 12 | `home_gk_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 34.5% |
| 13 | `away_gk_value` | lines | float64 | Transfermarkt appearances + player_valuations + players | 34.5% |
| 14 | `mkt_elo_diff` | market | float64 | Piyasa Elo'su (football-data kapanış oranları + tüm maç sonuçları) | 0.0% |
| 15 | `home_sot_share` | shots | float64 | football-data.co.uk şut istatistikleri | 50.4% |
| 16 | `away_sot_share` | shots | float64 | football-data.co.uk şut istatistikleri | 50.4% |
| 17 | `home_shot_share` | shots | float64 | football-data.co.uk şut istatistikleri | 50.4% |
| 18 | `away_shot_share` | shots | float64 | football-data.co.uk şut istatistikleri | 50.4% |

### Hesaplama tanımları

**`elo_diff`**

```text
elo(ev, D) + 55.0 * (1 - neutral) - elo(deplasman, D). elo(kulüp, D): kulübün tarihi D'den KESİN OLARAK önceki son maçından sonraki Elo rating'i (son maç D'den 400 günden eskiyse NaN). Maç kümesi: football-data.co.uk'teki 21 Avrupa 1. ligi + 6 adet 2. lig (['D2', 'E1', 'F2', 'I2', 'SC1', 'SP2']) + Transfermarkt'taki Ukrayna, Hırvatistan, Çekya, Sırbistan ligleri + UEFA Şampiyonlar Ligi / Avrupa Ligi / Konferans Ligi (ön eleme turları dahil). İç kupalar dahil DEĞİL. Elo motoru: veri 2008-07-01'de başlar, maçlar gün gün işlenir (aynı gündeki tüm maçlar gün başındaki ratinglerle hesaplanır, güncellemeler gün sonunda uygulanır). Beklenen skor E = 1 / (1 + 10^(-(R_ev - R_dep + 55.0*(1-neutral)) / 400)); S = 1 / 0.5 / 0. Değişim = K * u * p * G * (S - E); u = uefa_k_mult (UEFA maçıysa) yoksa 1; p = 2.0 (kulübün ilk 10 maçı) yoksa 1 (her kulüp için ayrı); G = 1 (|gol farkı| <= 1), 1.5 (= 2), (11 + |gol farkı|) / 8 (>= 3). K ve uefa_k_mult eğitimde ayarlanır ve şemadaki elo_params altında yazılıdır. Yeni kulüp başlangıcı: aynı ülkedeki mevcut ratinglerin %25 yüzdeliği (ülkesi bilinmeyen kulüpte ülkesi bilinmeyen kulüplerin havuzu); havuzda 6'dan az kulüp varsa 1500.0. Isınma: 2008-07-01 - 2014-07-01 arası maçlar 3 kez oynatılır (her tur bir öncekinin son ratingleriyle başlar), sonra tüm veri son kez işlenir. neutral: tarafsız saha ise 1.
```

**`home_form`**

```text
Ev sahibi takımın tarihi D'den KESİN OLARAK ÖNCEKİ son 5 maçındaki puan ortalaması (galibiyet 3, beraberlik 1, mağlubiyet 0; ev + deplasman; lig ve UEFA maçları, maç kümesi elo_diff ile aynı). Eğitimde takım bazında points.shift(1).rolling(5, min_periods=3).mean() ile birebir aynıdır. Pencerede 3'ten az maç varsa NaN. Ardışık iki maç arasında 200 günden uzun boşluk varsa pencere sıfırlanır; son maç D'den 200 günden eskiyse NaN. Değer aralığı 0-3.
```

**`away_form`**

```text
home_form ile aynı hesaplama, deplasman takımı için.
```

**`is_knockout`**

```text
UEFA ana turnuvasında (CL/EL/UECL) grup/lig aşamasından SONRAKİ eleme turundaysa 1.0 (play-off/intermediate stage, son 16, çeyrek final, yarı final, final); grup/lig aşaması, ön eleme turları ve iç lig maçlarında 0.0.
```

**`is_uefa`**

```text
Maç UEFA Şampiyonlar Ligi / Avrupa Ligi / Konferans Ligi maçıysa (ön eleme turları dahil) 1.0, iç lig maçıysa 0.0.
```

**`home_att_value`**

```text
log10(düzenli ilk 11'de players.position == 'Attack' olan oyuncuların v toplamı) - idx(T); bu mevkide değeri bilinen oyuncu yoksa NaN. (Oyuncu durumu tanımı: home_xi_value.)
```

**`away_att_value`**

```text
home_att_value ile aynı hesaplama, deplasman takımı için.
```

**`home_mid_value`**

```text
home_att_value ile aynı, players.position == 'Midfield' için.
```

**`away_mid_value`**

```text
home_mid_value ile aynı hesaplama, deplasman takımı için.
```

**`home_def_value`**

```text
home_att_value ile aynı, players.position == 'Defender' için.
```

**`away_def_value`**

```text
home_def_value ile aynı hesaplama, deplasman takımı için.
```

**`home_gk_value`**

```text
home_att_value ile aynı, players.position == 'Goalkeeper' için.
```

**`away_gk_value`**

```text
home_gk_value ile aynı hesaplama, deplasman takımı için.
```

**`mkt_elo_diff`**

```text
mkt(ev, D) + 55.0 * (1 - neutral) - mkt(deplasman, D). Piyasa Elo motoru elo_diff ile BİREBİR aynıdır (maç kümesi, gün bazlı güncelleme, ısınma, yeni kulüp, geçici K, bayatlık), tek fark değişim formülüdür: oranı olan maçlarda Değişim = K_m * u * p * [w * (E_piyasa - E) + (1 - w) * G * (S - E)], oranı olmayan maçlarda (UEFA, Transfermarkt ligleri) K_m * u * p * G * (S - E). E_piyasa = p_ev + 0.5 * p_beraberlik; p = (1/oran) / toplam(1/oran) (marj normalize). Oran: ['PSC', 'AvgC', 'B365C', 'Avg', 'BbAv', 'B365', 'PS'] sırasındaki ilk geçerli üçlü (kapanış oranları önce). K_m (mkt_k), w (mkt_w) eğitimde ayarlanır, u = uefa_k_mult (elo_diff ile aynı); şemadaki elo_params. Oranlar yalnızca maçtan SONRA rating güncellemek için kullanılır; özellik D'den önceki ratingdir.
```

**`home_sot_share`**

```text
Takımın tarihi D'den KESİN OLARAK önceki, şut verisi olan son 10 maçında toplam(isabetli şut) / (toplam(isabetli şut) + toplam(rakibin isabetli şutu)). Şut verili maç sayısı 5'ten azsa NaN; pencere sıfırlama ve bayatlık kuralı home_form ile aynı (boşluk ve bayatlık yalnızca şut verili maçlar üzerinden). UEFA maçlarında şut verisi yoktur.
```

**`away_sot_share`**

```text
home_sot_share ile aynı hesaplama, deplasman takımı için.
```

**`home_shot_share`**

```text
home_sot_share ile aynı, tüm şutlar (HS/AS) için.
```

**`away_shot_share`**

```text
home_shot_share ile aynı hesaplama, deplasman takımı için.
```

## Özellik grubu seçimi (kayan çapraz doğrulama)

İleri seçim: `base` grubundan başlanır; her turda CV ölçütünü (0.5 × log-loss tüm maçlar + 0.5 × log-loss UEFA) en çok iyileştiren grup eklenir; iyileşme 0.0003'ten küçükse durulur. Test seti kullanılmaz. Seçilen gruplar: **base, lines, shots, form, market**.

| Tur | Denenen grup | Ölçüt | Log-loss (tüm) | Log-loss (UEFA) | Eklendi |
|---|---|---|---|---|---|
| 0 | base | 0.96686 | 0.98809 | 0.94564 | ✔ |
| 1 | market | 0.96701 | 0.98800 | 0.94602 |  |
| 1 | shots | 0.96408 | 0.98460 | 0.94356 |  |
| 1 | form | 0.96619 | 0.98715 | 0.94523 |  |
| 1 | squad_value | 0.96268 | 0.98593 | 0.93942 |  |
| 1 | xi_value | 0.96291 | 0.98509 | 0.94073 |  |
| 1 | lines | 0.96244 | 0.98504 | 0.93984 | ✔ |
| 1 | production | 0.96631 | 0.98783 | 0.94478 |  |
| 1 | goals_form | 0.96662 | 0.98662 | 0.94663 |  |
| 1 | rest | 0.96651 | 0.98811 | 0.94490 |  |
| 1 | experience | 0.96666 | 0.98796 | 0.94537 |  |
| 1 | continuity | 0.96563 | 0.98801 | 0.94325 |  |
| 2 | market | 0.96248 | 0.98526 | 0.93970 |  |
| 2 | shots | 0.96183 | 0.98335 | 0.94031 | ✔ |
| 2 | form | 0.96242 | 0.98454 | 0.94029 |  |
| 2 | squad_value | 0.96256 | 0.98530 | 0.93982 |  |
| 2 | xi_value | 0.96280 | 0.98501 | 0.94059 |  |
| 2 | production | 0.96294 | 0.98569 | 0.94018 |  |
| 2 | goals_form | 0.96215 | 0.98386 | 0.94044 |  |
| 2 | rest | 0.96290 | 0.98523 | 0.94056 |  |
| 2 | experience | 0.96282 | 0.98582 | 0.93981 |  |
| 2 | continuity | 0.96273 | 0.98534 | 0.94012 |  |
| 3 | market | 0.96150 | 0.98349 | 0.93951 |  |
| 3 | form | 0.96146 | 0.98270 | 0.94021 | ✔ |
| 3 | squad_value | 0.96172 | 0.98316 | 0.94027 |  |
| 3 | xi_value | 0.96158 | 0.98278 | 0.94039 |  |
| 3 | production | 0.96199 | 0.98320 | 0.94078 |  |
| 3 | goals_form | 0.96157 | 0.98242 | 0.94072 |  |
| 3 | rest | 0.96174 | 0.98287 | 0.94062 |  |
| 3 | experience | 0.96215 | 0.98379 | 0.94051 |  |
| 3 | continuity | 0.96167 | 0.98310 | 0.94023 |  |
| 4 | market | 0.96099 | 0.98262 | 0.93936 | ✔ |
| 4 | squad_value | 0.96144 | 0.98262 | 0.94027 |  |
| 4 | xi_value | 0.96167 | 0.98225 | 0.94110 |  |
| 4 | production | 0.96187 | 0.98290 | 0.94083 |  |
| 4 | goals_form | 0.96155 | 0.98250 | 0.94061 |  |
| 4 | rest | 0.96139 | 0.98272 | 0.94007 |  |
| 4 | experience | 0.96141 | 0.98287 | 0.93995 |  |
| 4 | continuity | 0.96167 | 0.98283 | 0.94050 |  |
| 5 | squad_value | 0.96184 | 0.98279 | 0.94090 |  |
| 5 | xi_value | 0.96092 | 0.98214 | 0.93970 |  |
| 5 | production | 0.96144 | 0.98289 | 0.94000 |  |
| 5 | goals_form | 0.96114 | 0.98205 | 0.94024 |  |
| 5 | rest | 0.96139 | 0.98251 | 0.94026 |  |
| 5 | experience | 0.96073 | 0.98249 | 0.93898 |  |
| 5 | continuity | 0.96148 | 0.98265 | 0.94031 |  |

## Modelden çıkarılan özellikler

- `home_squad_value`: 'squad_value' grubu CV ileri seçiminde modele eklenmedi
- `away_squad_value`: 'squad_value' grubu CV ileri seçiminde modele eklenmedi
- `home_xi_value`: 'xi_value' grubu CV ileri seçiminde modele eklenmedi
- `away_xi_value`: 'xi_value' grubu CV ileri seçiminde modele eklenmedi
- `xi_value_diff`: 'xi_value' grubu CV ileri seçiminde modele eklenmedi
- `home_xi_star`: 'xi_value' grubu CV ileri seçiminde modele eklenmedi
- `away_xi_star`: 'xi_value' grubu CV ileri seçiminde modele eklenmedi
- `home_bench_value`: 'xi_value' grubu CV ileri seçiminde modele eklenmedi
- `away_bench_value`: 'xi_value' grubu CV ileri seçiminde modele eklenmedi
- `home_xi_ga90`: 'production' grubu CV ileri seçiminde modele eklenmedi
- `away_xi_ga90`: 'production' grubu CV ileri seçiminde modele eklenmedi
- `home_goals_for`: 'goals_form' grubu CV ileri seçiminde modele eklenmedi
- `away_goals_for`: 'goals_form' grubu CV ileri seçiminde modele eklenmedi
- `home_goals_against`: 'goals_form' grubu CV ileri seçiminde modele eklenmedi
- `away_goals_against`: 'goals_form' grubu CV ileri seçiminde modele eklenmedi
- `home_rest_days`: 'rest' grubu CV ileri seçiminde modele eklenmedi
- `away_rest_days`: 'rest' grubu CV ileri seçiminde modele eklenmedi
- `home_xi_age`: 'experience' grubu CV ileri seçiminde modele eklenmedi
- `away_xi_age`: 'experience' grubu CV ileri seçiminde modele eklenmedi
- `home_xi_uefa_apps`: 'experience' grubu CV ileri seçiminde modele eklenmedi
- `away_xi_uefa_apps`: 'experience' grubu CV ileri seçiminde modele eklenmedi
- `home_xi_last_match_share`: 'continuity' grubu CV ileri seçiminde modele eklenmedi
- `away_xi_last_match_share`: 'continuity' grubu CV ileri seçiminde modele eklenmedi

## Eğitimde ayarlanan Elo parametreleri

Izgara araması yalnızca 2021-07-01 öncesi maçlarla yapıldı (Elo beklenen skorunun ortalama kare hatası; iç lig ve UEFA eşit ağırlıklı). Tam tablo: `models/elo_tuning.csv`.

```json
{
  "k": 10.0,
  "uefa_k_mult": 3.0,
  "mkt_k": 10.0,
  "mkt_w": 0.5
}
```

## Sabit parametreler

```json
{
  "ELO_BASE": 1500.0,
  "HOME_ELO_ADVANTAGE": 55.0,
  "ELO_MAX_STALENESS_DAYS": 400,
  "ELO_PROVISIONAL_MATCHES": 10,
  "ELO_PROVISIONAL_K_MULT": 2.0,
  "ELO_NEWCOMER_PERCENTILE": 25,
  "ELO_NEWCOMER_MIN_POOL": 6,
  "ELO_BURN_IN_PASSES": 3,
  "ELO_BURN_IN_END": "2014-07-01",
  "ELO_DATA_START": "2008-07-01",
  "FORM_WINDOW": 5,
  "FORM_MIN_MATCHES": 3,
  "FORM_MAX_GAP_DAYS": 200,
  "SQUAD_WINDOW_DAYS": 180,
  "SQUAD_TOP_N": 16,
  "SQUAD_MIN_PLAYERS": 11,
  "VALUATION_MAX_AGE_DAYS": 540,
  "SQUAD_REFERENCE_COUNTRIES": [
    "ENG",
    "ESP",
    "GER",
    "ITA",
    "FRA",
    "NED",
    "POR",
    "BEL",
    "TUR",
    "GRE",
    "SCO",
    "RUS",
    "UKR",
    "DEN"
  ],
  "POINTS": {
    "win": 3,
    "draw": 1,
    "loss": 0
  },
  "GOALS_FORM_WINDOW": 10,
  "GOALS_FORM_MIN_MATCHES": 5,
  "PLAYER_WINDOW_MATCHES": 5,
  "XI_SIZE": 11,
  "BENCH_SIZE": 7,
  "XI_MIN_VALUED": 8,
  "BENCH_MIN_VALUED": 3,
  "PLAYER_STATE_MAX_GAP_DAYS": 200,
  "VALUE_INDEX_WINDOW_DAYS": 365,
  "PRODUCTION_WINDOW_DAYS": 365,
  "PRODUCTION_MIN_MINUTES": 900,
  "UEFA_EXPERIENCE_DAYS": 1095,
  "CONTINUITY_MIN_MINUTES": 45,
  "REST_DAYS_CAP": 30,
  "UEFA_MAIN_COMPETITION_IDS": [
    "CL",
    "EL",
    "UCOL"
  ],
  "ODDS_PRIORITY": [
    "PSC",
    "AvgC",
    "B365C",
    "Avg",
    "BbAv",
    "B365",
    "PS"
  ],
  "SHOTS_WINDOW": 10,
  "SHOTS_MIN_MATCHES": 5,
  "SECOND_TIER_LEAGUES": [
    "D2",
    "E1",
    "F2",
    "I2",
    "SC1",
    "SP2"
  ]
}
```

## Eksik değer politikası

- `elo_diff` NaN ise tahmin YAPILMAMALI (eğitimde bu satırlar atıldı).
- Diğer özellikler NaN olabilir: XGBoost bileşenleri NaN'ı kendi içinde işler; Lojistik Regresyon
  pipeline'ı eğitim medyanıyla doldurur. Doldurma işi pkl'ın içindedir, kullanan tarafta ayrıca
  doldurma YAPILMAMALI (NaN olduğu gibi verilmeli).

## Test metrikleri

| Alt küme | Model | n | Accuracy | Log-loss | Brier |
|---|---|---|---|---|---|
| tum_maclar | naive_class_frequencies | 8519 | 0.4371 | 1.0722 | 0.6487 |
| tum_maclar | xgb | 8519 | 0.5134 | 0.9959 | 0.5943 |
| tum_maclar | logreg | 8519 | 0.5107 | 0.9978 | 0.5956 |
| tum_maclar | poisson | 8519 | 0.5147 | 0.9960 | 0.5943 |
| tum_maclar | ensemble | 8519 | 0.5158 | 0.9963 | 0.5947 |
| tum_maclar | market_odds_reference | 6700 | 0.5204 | 0.9834 | 0.5861 |
| tum_maclar | ensemble_on_market_subset | 6700 | 0.5087 | 1.0021 | 0.5990 |
| ic_lig | naive_class_frequencies | 7592 | 0.4311 | 1.0760 | 0.6513 |
| ic_lig | xgb | 7592 | 0.5070 | 1.0010 | 0.5982 |
| ic_lig | logreg | 7592 | 0.5042 | 1.0035 | 0.5997 |
| ic_lig | poisson | 7592 | 0.5084 | 1.0012 | 0.5982 |
| ic_lig | ensemble | 7592 | 0.5088 | 1.0019 | 0.5988 |
| ic_lig | market_odds_reference | 6700 | 0.5204 | 0.9834 | 0.5861 |
| ic_lig | ensemble_on_market_subset | 6700 | 0.5087 | 1.0021 | 0.5990 |
| uefa_tumu | naive_class_frequencies | 927 | 0.4865 | 1.0408 | 0.6279 |
| uefa_tumu | xgb | 927 | 0.5663 | 0.9535 | 0.5632 |
| uefa_tumu | logreg | 927 | 0.5642 | 0.9510 | 0.5624 |
| uefa_tumu | poisson | 927 | 0.5663 | 0.9530 | 0.5627 |
| uefa_tumu | ensemble | 927 | 0.5728 | 0.9505 | 0.5614 |
| uefa_ana_turnuva | naive_class_frequencies | 528 | 0.5000 | 1.0340 | 0.6232 |
| uefa_ana_turnuva | xgb | 528 | 0.5795 | 0.9398 | 0.5533 |
| uefa_ana_turnuva | logreg | 528 | 0.5833 | 0.9362 | 0.5517 |
| uefa_ana_turnuva | poisson | 528 | 0.5814 | 0.9390 | 0.5531 |
| uefa_ana_turnuva | ensemble | 528 | 0.5890 | 0.9349 | 0.5502 |
| uefa_eleme_turu | naive_class_frequencies | 132 | 0.4621 | 1.0418 | 0.6301 |
| uefa_eleme_turu | xgb | 132 | 0.5379 | 1.0016 | 0.5950 |
| uefa_eleme_turu | logreg | 132 | 0.5152 | 1.0016 | 0.5988 |
| uefa_eleme_turu | poisson | 132 | 0.5530 | 0.9982 | 0.5954 |
| uefa_eleme_turu | ensemble | 132 | 0.5379 | 0.9994 | 0.5948 |

`ensemble` = kaydedilen modelin çıktısı. `market_odds_reference`: kapanış oranlarından türetilen olasılıklar (yalnızca iç lig; kıyas ölçütüdür, modelde kullanılmaz).

Kalibrasyon (ensemble, test; ECE = beklenen kalibrasyon hatası, 0 ideal):

- tum_maclar: away_win=0.0137, draw=0.0136, home_win=0.0200, top_label=0.0232
- uefa_tumu: away_win=0.0407, draw=0.0202, home_win=0.0331, top_label=0.0212

## Kütüphane versiyonları

| Paket | Versiyon |
|---|---|
| python | 3.13.5 |
| xgboost | 3.4.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| pandas | 3.0.5 |
| numpy | 2.5.3 |
| joblib | 1.6.0 |

## Makine tarafından okunan sözleşme

<!-- SCHEMA_JSON_BEGIN -->
```json
{
  "schema_version": 3,
  "contract_hash": "0589222bbcf9097f483ac2d818772ff33373ce8c81e9ef3949df6684bff524cd",
  "feature_columns": [
    "elo_diff",
    "home_form",
    "away_form",
    "is_knockout",
    "is_uefa",
    "home_att_value",
    "away_att_value",
    "home_mid_value",
    "away_mid_value",
    "home_def_value",
    "away_def_value",
    "home_gk_value",
    "away_gk_value",
    "mkt_elo_diff",
    "home_sot_share",
    "away_sot_share",
    "home_shot_share",
    "away_shot_share"
  ],
  "features": [
    {
      "name": "elo_diff",
      "dtype": "float64",
      "source": "Kendi ligler arası Elo (football-data + Transfermarkt maç sonuçları)",
      "definition": "elo(ev, D) + 55.0 * (1 - neutral) - elo(deplasman, D). elo(kulüp, D): kulübün tarihi D'den KESİN OLARAK önceki son maçından sonraki Elo rating'i (son maç D'den 400 günden eskiyse NaN). Maç kümesi: football-data.co.uk'teki 21 Avrupa 1. ligi + 6 adet 2. lig (['D2', 'E1', 'F2', 'I2', 'SC1', 'SP2']) + Transfermarkt'taki Ukrayna, Hırvatistan, Çekya, Sırbistan ligleri + UEFA Şampiyonlar Ligi / Avrupa Ligi / Konferans Ligi (ön eleme turları dahil). İç kupalar dahil DEĞİL. Elo motoru: veri 2008-07-01'de başlar, maçlar gün gün işlenir (aynı gündeki tüm maçlar gün başındaki ratinglerle hesaplanır, güncellemeler gün sonunda uygulanır). Beklenen skor E = 1 / (1 + 10^(-(R_ev - R_dep + 55.0*(1-neutral)) / 400)); S = 1 / 0.5 / 0. Değişim = K * u * p * G * (S - E); u = uefa_k_mult (UEFA maçıysa) yoksa 1; p = 2.0 (kulübün ilk 10 maçı) yoksa 1 (her kulüp için ayrı); G = 1 (|gol farkı| <= 1), 1.5 (= 2), (11 + |gol farkı|) / 8 (>= 3). K ve uefa_k_mult eğitimde ayarlanır ve şemadaki elo_params altında yazılıdır. Yeni kulüp başlangıcı: aynı ülkedeki mevcut ratinglerin %25 yüzdeliği (ülkesi bilinmeyen kulüpte ülkesi bilinmeyen kulüplerin havuzu); havuzda 6'dan az kulüp varsa 1500.0. Isınma: 2008-07-01 - 2014-07-01 arası maçlar 3 kez oynatılır (her tur bir öncekinin son ratingleriyle başlar), sonra tüm veri son kez işlenir. neutral: tarafsız saha ise 1.",
      "group": "base"
    },
    {
      "name": "home_form",
      "dtype": "float64",
      "source": "football-data.co.uk + Transfermarkt maç sonuçları",
      "definition": "Ev sahibi takımın tarihi D'den KESİN OLARAK ÖNCEKİ son 5 maçındaki puan ortalaması (galibiyet 3, beraberlik 1, mağlubiyet 0; ev + deplasman; lig ve UEFA maçları, maç kümesi elo_diff ile aynı). Eğitimde takım bazında points.shift(1).rolling(5, min_periods=3).mean() ile birebir aynıdır. Pencerede 3'ten az maç varsa NaN. Ardışık iki maç arasında 200 günden uzun boşluk varsa pencere sıfırlanır; son maç D'den 200 günden eskiyse NaN. Değer aralığı 0-3.",
      "group": "form"
    },
    {
      "name": "away_form",
      "dtype": "float64",
      "source": "football-data.co.uk + Transfermarkt maç sonuçları",
      "definition": "home_form ile aynı hesaplama, deplasman takımı için.",
      "group": "form"
    },
    {
      "name": "is_knockout",
      "dtype": "float64",
      "source": "Transfermarkt games.round",
      "definition": "UEFA ana turnuvasında (CL/EL/UECL) grup/lig aşamasından SONRAKİ eleme turundaysa 1.0 (play-off/intermediate stage, son 16, çeyrek final, yarı final, final); grup/lig aşaması, ön eleme turları ve iç lig maçlarında 0.0.",
      "group": "base"
    },
    {
      "name": "is_uefa",
      "dtype": "float64",
      "source": "maç kaynağı (Transfermarkt competition_id)",
      "definition": "Maç UEFA Şampiyonlar Ligi / Avrupa Ligi / Konferans Ligi maçıysa (ön eleme turları dahil) 1.0, iç lig maçıysa 0.0.",
      "group": "base"
    },
    {
      "name": "home_att_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "log10(düzenli ilk 11'de players.position == 'Attack' olan oyuncuların v toplamı) - idx(T); bu mevkide değeri bilinen oyuncu yoksa NaN. (Oyuncu durumu tanımı: home_xi_value.)",
      "group": "lines"
    },
    {
      "name": "away_att_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "home_att_value ile aynı hesaplama, deplasman takımı için.",
      "group": "lines"
    },
    {
      "name": "home_mid_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "home_att_value ile aynı, players.position == 'Midfield' için.",
      "group": "lines"
    },
    {
      "name": "away_mid_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "home_mid_value ile aynı hesaplama, deplasman takımı için.",
      "group": "lines"
    },
    {
      "name": "home_def_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "home_att_value ile aynı, players.position == 'Defender' için.",
      "group": "lines"
    },
    {
      "name": "away_def_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "home_def_value ile aynı hesaplama, deplasman takımı için.",
      "group": "lines"
    },
    {
      "name": "home_gk_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "home_att_value ile aynı, players.position == 'Goalkeeper' için.",
      "group": "lines"
    },
    {
      "name": "away_gk_value",
      "dtype": "float64",
      "source": "Transfermarkt appearances + player_valuations + players",
      "definition": "home_gk_value ile aynı hesaplama, deplasman takımı için.",
      "group": "lines"
    },
    {
      "name": "mkt_elo_diff",
      "dtype": "float64",
      "source": "Piyasa Elo'su (football-data kapanış oranları + tüm maç sonuçları)",
      "definition": "mkt(ev, D) + 55.0 * (1 - neutral) - mkt(deplasman, D). Piyasa Elo motoru elo_diff ile BİREBİR aynıdır (maç kümesi, gün bazlı güncelleme, ısınma, yeni kulüp, geçici K, bayatlık), tek fark değişim formülüdür: oranı olan maçlarda Değişim = K_m * u * p * [w * (E_piyasa - E) + (1 - w) * G * (S - E)], oranı olmayan maçlarda (UEFA, Transfermarkt ligleri) K_m * u * p * G * (S - E). E_piyasa = p_ev + 0.5 * p_beraberlik; p = (1/oran) / toplam(1/oran) (marj normalize). Oran: ['PSC', 'AvgC', 'B365C', 'Avg', 'BbAv', 'B365', 'PS'] sırasındaki ilk geçerli üçlü (kapanış oranları önce). K_m (mkt_k), w (mkt_w) eğitimde ayarlanır, u = uefa_k_mult (elo_diff ile aynı); şemadaki elo_params. Oranlar yalnızca maçtan SONRA rating güncellemek için kullanılır; özellik D'den önceki ratingdir.",
      "group": "market"
    },
    {
      "name": "home_sot_share",
      "dtype": "float64",
      "source": "football-data.co.uk şut istatistikleri",
      "definition": "Takımın tarihi D'den KESİN OLARAK önceki, şut verisi olan son 10 maçında toplam(isabetli şut) / (toplam(isabetli şut) + toplam(rakibin isabetli şutu)). Şut verili maç sayısı 5'ten azsa NaN; pencere sıfırlama ve bayatlık kuralı home_form ile aynı (boşluk ve bayatlık yalnızca şut verili maçlar üzerinden). UEFA maçlarında şut verisi yoktur.",
      "group": "shots"
    },
    {
      "name": "away_sot_share",
      "dtype": "float64",
      "source": "football-data.co.uk şut istatistikleri",
      "definition": "home_sot_share ile aynı hesaplama, deplasman takımı için.",
      "group": "shots"
    },
    {
      "name": "home_shot_share",
      "dtype": "float64",
      "source": "football-data.co.uk şut istatistikleri",
      "definition": "home_sot_share ile aynı, tüm şutlar (HS/AS) için.",
      "group": "shots"
    },
    {
      "name": "away_shot_share",
      "dtype": "float64",
      "source": "football-data.co.uk şut istatistikleri",
      "definition": "home_shot_share ile aynı hesaplama, deplasman takımı için.",
      "group": "shots"
    }
  ],
  "params": {
    "ELO_BASE": 1500.0,
    "HOME_ELO_ADVANTAGE": 55.0,
    "ELO_MAX_STALENESS_DAYS": 400,
    "ELO_PROVISIONAL_MATCHES": 10,
    "ELO_PROVISIONAL_K_MULT": 2.0,
    "ELO_NEWCOMER_PERCENTILE": 25,
    "ELO_NEWCOMER_MIN_POOL": 6,
    "ELO_BURN_IN_PASSES": 3,
    "ELO_BURN_IN_END": "2014-07-01",
    "ELO_DATA_START": "2008-07-01",
    "FORM_WINDOW": 5,
    "FORM_MIN_MATCHES": 3,
    "FORM_MAX_GAP_DAYS": 200,
    "SQUAD_WINDOW_DAYS": 180,
    "SQUAD_TOP_N": 16,
    "SQUAD_MIN_PLAYERS": 11,
    "VALUATION_MAX_AGE_DAYS": 540,
    "SQUAD_REFERENCE_COUNTRIES": [
      "ENG",
      "ESP",
      "GER",
      "ITA",
      "FRA",
      "NED",
      "POR",
      "BEL",
      "TUR",
      "GRE",
      "SCO",
      "RUS",
      "UKR",
      "DEN"
    ],
    "POINTS": {
      "win": 3,
      "draw": 1,
      "loss": 0
    },
    "GOALS_FORM_WINDOW": 10,
    "GOALS_FORM_MIN_MATCHES": 5,
    "PLAYER_WINDOW_MATCHES": 5,
    "XI_SIZE": 11,
    "BENCH_SIZE": 7,
    "XI_MIN_VALUED": 8,
    "BENCH_MIN_VALUED": 3,
    "PLAYER_STATE_MAX_GAP_DAYS": 200,
    "VALUE_INDEX_WINDOW_DAYS": 365,
    "PRODUCTION_WINDOW_DAYS": 365,
    "PRODUCTION_MIN_MINUTES": 900,
    "UEFA_EXPERIENCE_DAYS": 1095,
    "CONTINUITY_MIN_MINUTES": 45,
    "REST_DAYS_CAP": 30,
    "UEFA_MAIN_COMPETITION_IDS": [
      "CL",
      "EL",
      "UCOL"
    ],
    "ODDS_PRIORITY": [
      "PSC",
      "AvgC",
      "B365C",
      "Avg",
      "BbAv",
      "B365",
      "PS"
    ],
    "SHOTS_WINDOW": 10,
    "SHOTS_MIN_MATCHES": 5,
    "SECOND_TIER_LEAGUES": [
      "D2",
      "E1",
      "F2",
      "I2",
      "SC1",
      "SP2"
    ]
  },
  "elo_params": {
    "k": 10.0,
    "uefa_k_mult": 3.0,
    "mkt_k": 10.0,
    "mkt_w": 0.5
  },
  "ensemble": {
    "weights": {
      "xgb": 0.45,
      "logreg": 0.25,
      "poisson": 0.3
    },
    "rho": -0.030443580996439037,
    "temperature": 0.9192906859117506,
    "max_goals": 10
  },
  "class_order": [
    "home_win",
    "draw",
    "away_win"
  ],
  "component_params": {
    "xgb": {
      "learning_rate": 0.03,
      "max_depth": 5,
      "min_child_weight": 100,
      "colsample_bytree": 0.8,
      "reg_lambda": 2.0,
      "half_life": 4.0,
      "n_estimators": 232
    },
    "logreg": {
      "C": 1.0,
      "half_life": 4.0
    },
    "poisson": {
      "learning_rate": 0.03,
      "max_depth": 4,
      "min_child_weight": 50,
      "colsample_bytree": 0.8,
      "reg_lambda": 2.0,
      "half_life": 4.0,
      "n_estimators": 328
    }
  },
  "excluded_features": [
    {
      "name": "home_squad_value",
      "reason": "'squad_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_squad_value",
      "reason": "'squad_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_xi_value",
      "reason": "'xi_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_xi_value",
      "reason": "'xi_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "xi_value_diff",
      "reason": "'xi_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_xi_star",
      "reason": "'xi_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_xi_star",
      "reason": "'xi_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_bench_value",
      "reason": "'xi_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_bench_value",
      "reason": "'xi_value' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_xi_ga90",
      "reason": "'production' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_xi_ga90",
      "reason": "'production' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_goals_for",
      "reason": "'goals_form' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_goals_for",
      "reason": "'goals_form' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_goals_against",
      "reason": "'goals_form' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_goals_against",
      "reason": "'goals_form' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_rest_days",
      "reason": "'rest' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_rest_days",
      "reason": "'rest' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_xi_age",
      "reason": "'experience' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_xi_age",
      "reason": "'experience' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_xi_uefa_apps",
      "reason": "'experience' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_xi_uefa_apps",
      "reason": "'experience' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "home_xi_last_match_share",
      "reason": "'continuity' grubu CV ileri seçiminde modele eklenmedi"
    },
    {
      "name": "away_xi_last_match_share",
      "reason": "'continuity' grubu CV ileri seçiminde modele eklenmedi"
    }
  ],
  "trained_at": "2026-09-21 06:47:02",
  "data_range": {
    "development": {
      "from": "2014-07-01",
      "to": "2025-06-30",
      "n": 73281,
      "n_uefa": 7736
    },
    "test": {
      "from": "2025-07-01",
      "to": "2026-09-17",
      "n": 8519,
      "n_uefa": 927
    }
  },
  "fitted_on": "geliştirme + test (--refit-all; test metrikleri bir önceki fit'e ait)",
  "data_end": "2026-09-17",
  "library_versions": {
    "python": "3.13.5",
    "xgboost": "3.4.1",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1",
    "pandas": "3.0.5",
    "numpy": "2.5.3",
    "joblib": "1.6.0"
  }
}
```
<!-- SCHEMA_JSON_END -->
