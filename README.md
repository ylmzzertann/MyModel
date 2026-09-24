# futbol-ml-modeli

Geçmiş veriyle eğitilmiş maç sonucu modeli.

- **Olasılık tahmini:** ev sahibi kazanır / beraberlik / deplasman kazanır.
- **Skor dağılımı:** simülasyonun gol üretmesi için tam skor olasılıkları.

Bağımsız bir projedir. Başka bir uygulamaya taşınacak parçalar `models/mac_modeli.pkl` ve `feature_schema.md` dosyalarıdır.

```
data/                   ham veriler (indirilen) + team_aliases.csv (elle bakılan eşleştirmeler)
src/data_collection.py  veri indirme / yükleme
src/features.py         takım kimliği, Elo, piyasa Elo'su, form, şut payı, oyuncu bazlı takım gücü
                        (eğitim ve tahmin AYNI kodu kullanır)
src/train.py            kayan çapraz doğrulama, özellik seçimi, hiperparametre araması, topluluk, feature_schema.md
src/predict.py          MatchPredictor (predict, predict_score, predict_many) + topluluk formülü
src/export_predictions.py  simülasyon için tahmin tablosu dışa aktarımı
exports/                sim_predictions_<tarih>.json, sim_score_sampler.py (simülasyon tarafı referans kod)
models/                 mac_modeli.pkl, metrics.json, calibration_report.csv, elo_tuning.csv,
                        feature_selection.csv, hyperparameter_search.csv
feature_schema.md       modelin girdi/çıktı sözleşmesi (otomatik üretilir)
```

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Tüm komutlar proje kökünden `python -m src.<modül>` şeklinde çalıştırılır.

## Veri kaynakları

| Kaynak | İçerik | Nereye |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | 21 Avrupa 1. ligi + 6 adet 2. lig (Championship, İskoçya 2, 2. Bundesliga, Serie B, Segunda, Ligue 2), 2008-09'dan bugüne: skor, **kapanış oranları**, **şut / isabetli şut**, kart, faul | `data/football_data/` |
| [Transfermarkt veri seti](https://github.com/dcaribou/transfermarkt-datasets) (CC0) | UEFA CL/EL/UECL ve ön eleme maçları, Ukrayna/Hırvatistan/Çekya/Sırbistan ligleri, oyuncu maç kadroları, piyasa değeri geçmişi, oyuncu mevkileri | `data/transfermarkt/` |

İkisi de kayıt gerekmeden `python -m src.data_collection` ile indirilir.

### Kullanılmayan kaynaklar ve nedenleri

| Kaynak | Neden |
|---|---|
| ClubElo | `api.clubelo.com` rating endpoint'leri geliştirme boyunca hep 502 döndü. Ligler arası Elo kendimiz hesaplanıyor. |
| Understat (xG) | `robots.txt` tüm botları yasaklıyor (`Disallow: /`). Veri kazınmadı. |
| FBref | Botlara 403 döndürüyor, kullanım şartları toplu veri çekmeyi kısıtlıyor. Opta istatistikleri de Ocak 2026'da kaldırıldı. |
| Opta / StatsBomb / Wyscout | Yalnızca kurumsal lisansla satılıyor. |
| API-Football (sakatlık, kesin kadro) | API anahtarı gerekiyor. En değerli eksik bilgi bu (bkz. "Sonraki adımlar"). |

**Transfermarkt lisansı:** veri seti CC0, ancak içerik Transfermarkt sitesinden toplanmış. Ticari kullanımda Transfermarkt'ın kullanım şartlarını ayrıca kontrol edin.

### Takım kimliği

Kanonik anahtar Transfermarkt kulüp id'sidir (`tm:131` = FC Barcelona).

- **Maç çakıştırma:** football-data adları bu id'lere, 1. lig maçlarında aynı gün aynı skorla oynanmış maçlar çakıştırılarak eşlenir.
- **Ad benzerliği:** kalan adlar ülke içinde ad benzerliğiyle eşlenir. İç ligi Transfermarkt'ta 2012'den beri olan ülkelerde "ülkesi bilinmeyen" kulüp havuzu kullanılmaz. Bu kural, "Lincoln" → "Lincoln Red Imps" gibi hataları önler.
- **Eşlenemeyenler:** `fd:ÜLKE:ad` anahtarıyla ayrı kimlik olarak kalır. Çoğu, hiç 1. ligde oynamamış 2. lig kulübüdür.
- **Güvenlik:** aynı kulübün aynı gün iki maçı olamaz. Böyle bir durum çıkarsa ilgili maçlar atılır ve uyarı verilir.
- **Rapor ve düzeltme:** rapor `data/reports/team_identity.csv` dosyasına yazılır. Düzeltmeler `data/team_aliases.csv` dosyasına eklenir.

## Özellikler

Tüm özellikler maç tarihinden **kesin olarak önceki** bilgiyle hesaplanır. Tanımların tamamı `feature_schema.md`'de.

| Grup | Ne ölçer | Modelde |
|---|---|---|
| `base` | Ligler arası Elo farkı (ev avantajı dahil), eleme turu mu, UEFA maçı mı | ✔ |
| `lines` | Düzenli ilk 11'in (son 5 maçta en çok süre alanlar) hücum / orta saha / defans / kaleci hatlarındaki piyasa değeri | ✔ |
| `shots` | Son 10 maçta isabetli şut payı ve şut payı (xG'nin erişilebilir yerine geçeni) | ✔ |
| `form` | Son 5 maçın puan ortalaması | ✔ |
| `market` | **Piyasa Elo'su:** oranı olan maçlarda güncelleme, sonuç ile kapanış oranlarının ima ettiği beklenen skorun karışımı. Oranlar yalnızca geçmiş maçlardan rating öğrenmek için kullanılır; simülasyonda da çalışır. | ✔ |
| `xi_value`, `squad_value` | İlk 11 toplam değeri, yıldız, yedek derinliği; 180 günlük kadro değeri | hat değerlerinin yanında ek bilgi getirmedi |
| `experience`, `continuity`, `rest`, `production`, `goals_form` | Yaş/UEFA tecrübesi, rotasyon, dinlenme, gol+asist üretimi, gol ortalamaları | eşiği geçemedi |

### Elo motorları

- **Maç kümesi ve sıra:** iç lig, 2. lig ve UEFA maçları tarih sırasıyla, gün gün işlenir. Aynı gündeki maçlar gün başındaki ratinglerle hesaplanır.
- **Ligler arası akış:** ligler arası güç farkı UEFA maçları üzerinden aktarılır.
- **Ayarlanan parametreler:** K, UEFA maçı K çarpanı, piyasa Elo'sunun K'si ve oran ağırlığı `w`. Hepsi **yalnızca ilk doğrulama sezonundan önceki** veriyle ayarlanır.
- **Son değerler:** K=10, UEFA çarpanı=3, piyasa K=10, w=0,5.
- **Isınma dönemi:** 2008-2014 arası maçlar 3 kez oynatılır.

## Eğitim

```bash
python -m src.data_collection        # football-data + Transfermarkt indir / güncelle
python -m src.train                  # tam eğitim (~50 dk)
python -m src.train --reuse-dataset  # özellik tablosu değişmediyse önbellekten
python -m src.train --refit-all      # testten sonra test dönemini de dahil edip yeniden eğit
```

1. **Zaman bazlı ayrım.**
   - Test: son tamamlanmış sezonun başından bugüne (şu an 2025-07-01 →).
   - Geliştirme: 2014-07-01'den teste kadar.
   - Eğitim satırları 1. lig ve UEFA maçlarıdır; 2. lig maçları yalnızca geçmiş özellikleri besler. Rastgele ayrım hiçbir yerde yok.
2. **Kayan çapraz doğrulama.** Geliştirme dönemindeki son 4 sezon (2021-22 → 2024-25) sırayla doğrulama sezonu olur. Model yalnızca o sezondan önceki verilerle eğitilir. Tüm kararlar 4 sezonun fold dışı tahminleriyle verilir; tek sezonluk doğrulamanın gürültüsü bu sayede azalır.
   - **Ölçüt:** `0.5 × log-loss(tüm maçlar) + 0.5 × log-loss(UEFA maçları)`.
3. **Karar adımları.**
   - Özellik grubu ileri seçimi.
   - XGBoost hiperparametre araması, yakın sezonlara ağırlık veren yarı ömür seçeneği dahil.
   - Lojistik Regresyon.
   - Poisson skor modelleri ve Dixon-Coles ρ.
   - Topluluk ağırlıkları ve sıcaklık kalibrasyonu.
4. **Topluluk formülü.** Her maç için üç bileşenin G/B/M olasılıkları birleştirilir:
   ```
   p = w_xgb·XGBoost + w_logreg·LojistikRegresyon + w_poisson·G/B/M(Poisson×Dixon-Coles skor matrisi)
   p_final = softmax(log p / T)
   ```
5. **Kontroller.** Eğitimden önce aşağıdaki kontroller çalışır; biri geçmezse eğitim durur.
   - Form, gol formu ve şut payı, bağımsız yazılmış `shift(1)` hesabıyla aynı olmalı.
   - Tarihle sorgulanan Elo ve piyasa Elo'su, motorun maç öncesi rating'iyle aynı olmalı.
   - Oyuncu bazlı özellikler için 300 rastgele maç, bağımsız olarak yeniden hesaplanır ve aynı çıkmalı.

## Son eğitimin sonuçları (2026-09-14)

### Çapraz doğrulama: hangi bilgi işe yarıyor?

Aşağıda 1. tur sonuçları var: her grup yalnızca `base` grubunun üzerine eklendi. 4 doğrulama sezonunun toplamı; düşük olan daha iyi.

| Grup | Ölçüt | Log-loss (tüm) | Log-loss (UEFA) |
|---|---|---|---|
| `base` | 0.96686 | 0.98809 | 0.94564 |
| + `lines` | **0.96244** | 0.98504 | 0.93984 |
| + `squad_value` | 0.96268 | 0.98593 | 0.93942 |
| + `xi_value` | 0.96291 | 0.98509 | 0.94073 |
| + `shots` | 0.96408 | 0.98460 | 0.94356 |
| + `continuity` | 0.96563 | 0.98801 | 0.94325 |
| + `form` | 0.96619 | 0.98715 | 0.94523 |
| + `production` | 0.96631 | 0.98783 | 0.94478 |
| + `rest` | 0.96651 | 0.98811 | 0.94490 |
| + `goals_form` | 0.96662 | 0.98662 | 0.94663 |
| + `experience` | 0.96666 | 0.98796 | 0.94537 |
| + `market` | 0.96701 | 0.98800 | 0.94602 |

- **Seçim yolu:** `base` → +`lines` (0.96244) → +`shots` (0.96183) → +`form` (0.96146) → +`market` (0.96099).
- **Piyasa Elo'su tek başına işe yaramadı, ama diğer bilgilerle birlikte katkı sağladı.** Tek başına normal Elo ile neredeyse aynı bilgiyi taşıyor; oyuncu ve şut bilgisi eklenince tamamlayıcı bir sinyal olarak işe yarıyor.
- **Gol tabanlı özellikler (`goals_form`, `production`) UEFA'da zarar veriyor.** Gol sayıları lig gücüne göre ölçeklenmiyor. Şut payı ise oran olduğu için ligden bağımsız ve işe yarıyor.

| Aşama (CV) | Ölçüt |
|---|---|
| Yalnız `base` | 0.96686 |
| Seçilen özellikler, varsayılan XGBoost | 0.96120 |
| + hiperparametre araması (derinlik 5, `min_child_weight` 100, yarı ömür 4 yıl) | 0.96060 |
| Poisson skor modeli tek başına (ρ = -0.030) | 0.96058 |
| **Topluluk** (XGBoost 0.45, LogReg 0.25, Poisson 0.30; T = 0.919) | **0.95893** |

### Test (2025-07-01 → 2026-09-10, eğitimde hiç kullanılmadı)

| Alt küme | n | Topluluk accuracy | Topluluk log-loss | Brier | Önceki model log-loss |
|---|---|---|---|---|---|
| Tüm maçlar | 8344 | 0.516 | 0.9965 | 0.5948 | 0.9971 |
| UEFA (ön eleme dahil) | 927 | 0.573 | 0.9505 | 0.5614 | 0.9516 |
| UEFA ana turnuva (CL/EL/UECL) | 528 | **0.589** | **0.9349** | 0.5502 | 0.9425 |
| UEFA eleme turu | 132 | 0.538 | 0.9994 | 0.5948 | 0.9938 |
| *Kıyas: sınıf frekansları* | 8344 | 0.437 | 1.0722 | 0.6487 | |
| *Kıyas: kapanış oranları (iç lig)* | 6525 | 0.521 | 0.9839 | 0.5864 | |
| *Topluluk, aynı iç lig maçlarında* | 6525 | 0.509 | 1.0024 | 0.5992 | |

Bileşenler testte tek tek:

| Bileşen | Tüm maçlar | UEFA |
|---|---|---|
| XGBoost | 0.9960 | 0.9535 |
| Poisson | 0.9961 | 0.9530 |
| LogReg | 0.9980 | 0.9510 |

"Önceki model", bir önceki adımdaki oyuncu bazlı XGBoost modelidir.

**Dürüst değerlendirme:**
- **Kazanç UEFA maçlarında belirgin.** UEFA ana turnuva maçlarında log-loss 0.0076 düştü. Tüm maçlarda iyileşme küçük: 0.0006. 132 maçlık eleme turu alt kümesi gürültülü ve orada biraz kötüleşti.
- **Kalibrasyon:** 4 çapraz doğrulama sezonunun **hepsinde** model hafif çekingendi; sıcaklık 0.88-0.96 çıktı. Bu yüzden olasılıklar biraz keskinleştirildi. Test sezonu ise ters davrandı: model orada hafif fazla güvenli (tüm maçlarda ECE 0.023). İç lig testinde topluluk, sıcaklık yüzünden tek başına XGBoost'tan biraz geride kaldı.
  - Bunun yöntem kaynaklı olup olmadığı yalnızca CV verisiyle kontrol edildi: doğrulama katmanlarındaki modeller son modelle aynı şekilde eğitilince de sıcaklık 1'in altında kaldı.
  - Sıcaklığı test sezonuna bakarak değiştirmek test sonucunu yanıltıcı yapacağı için yöntem değiştirilmedi.
- **Piyasa farkı hâlâ var:** iç lig maçlarında kapanış oranlarının 0.0185 log-loss gerisinde. Kalan farkın büyük kısmı, modelde olmayan sakatlık, kesin kadro ve haber bilgisi.

## Tahmin

```python
from src.predict import MatchPredictor

predictor = MatchPredictor()
predictor.predict("Galatasaray", "Bayern Munich", "2026-10-21")
# {"home_win": 0.200, "draw": 0.204, "away_win": 0.596}

predictor.predict_score("Galatasaray", "Bayern Munich", "2026-10-21")
# {"expected_goals": {"home": 1.17, "away": 1.99},
#  "score_matrix": 11x11 olasılık ([i][j] = ev i - deplasman j),
#  "top_scores": [{"score": "1-2", "probability": 0.105}, {"score": "1-1", ...}, ...],
#  "wdl": predict() ile aynı}

predictor.predict("Arsenal FC", "Inter Milan", "2027-05-29", neutral=True, is_knockout=True)  # final
predictor.features_for("Galatasaray", "Bayern Munich", "2026-10-21")  # modelin gördüğü girdiyi incele
```

- **Skor matrisi:** Poisson ve Dixon-Coles'tan gelir. Ev kazanır / beraberlik / deplasman bölgelerinin toplamı `predict()` olasılıklarına eşit olacak şekilde ölçeklenir; yani skor simülasyonu ile G/B/M tahmini tutarlıdır. Uzatma için yaklaşık `λ × 30/90` kullanılabilir; model uzatmalarla eğitilmedi.
- **Takım adı:** Transfermarkt adı, football-data adı ya da doğrudan anahtar (`tm:141`) verilebilir. Belirsiz ad hata ile adayları listeler.
- **`is_uefa`:** verilmezse çıkarılır. Eleme turu ise ya da ülkeler farklıysa UEFA maçı sayılır; aynı ülkenin iki kulübüyse lig maçı sayılır.
- **`overrides`:** veriden hesaplanan bir özelliği dışarıdan verilen değerle değiştirir, ör. simülasyon kendi kadrosundan hat değerlerini verebilir. Yalnızca şemadaki adlar kabul edilir; `elo_diff`, `mkt_elo_diff`, `is_knockout`, `is_uefa` bu yolla değiştirilemez.

## Üretim modeli ve simülasyon için dışa aktarım

- **`models/mac_modeli.pkl` üretim modelidir.** Test sonuçları alındıktan sonra, aynı kararlarla ve test dönemi de dahil tüm veriyle yeniden eğitildi (`--refit-all`): aynı özellikler, hiperparametreler ve topluluk ağırlıkları. Test metrikleri bu sürüme değil, bir önceki fit'e aittir. Test edilen sürüm `models/test_edilen_model/` klasöründe saklanıyor.
- **Tahmin tablosu dışa aktarımı:**
  ```bash
  python -m src.export_predictions --teams-dir "<simülasyon>/API/data" --date 2026-09-14
  ```
  Simülasyonun takım listelerindeki her turnuva için tüm ev/deplasman eşleşmelerinin G/B/M olasılıklarını ve beklenen gollerini `exports/sim_predictions_<tarih>.json` dosyasına yazar. Üç maç türü var: lig, eleme, final.
- **Takım eşleştirmesi:** `data/sim_team_mapping.csv` dosyasında tutulur. Eşlenemeyen takım olursa tablo yazılmaz.
- **Entegrasyon adımları:** [ENTEGRASYON.md](ENTEGRASYON.md). Skor örnekleme referans kodu `exports/sim_score_sampler.py` dosyasında.

## Haftalık otomatik güncelleme

Windows Görev Zamanlayıcı'daki **"futbol-ml-modeli haftalik guncelleme"** görevi her **pazartesi 10:00**'da `run_weekly.bat` dosyasını çalıştırır (`python -m src.weekly_update`).
- Bilgisayar o saatte kapalıysa, açıldığında çalışır.
- Düşük öncelikle çalışır.
- Yalnızca kullanıcı oturumu açıkken çalışır.

| Adım | Ne yapar | Süre |
|---|---|---|
| 1. Veri | football-data'dan devam eden sezonu ve extra ligleri indirir. Transfermarkt'ta yeni sürüm varsa onu da indirir. | ~1 dk |
| 2. Özellikler | Elo, piyasa Elo'su, form, şut payı ve oyuncu durumu yeni maçlarla yeniden hesaplanır. **Model değişmez**; tahminler bu özelliklerle güncellenir. | ~1 dk |
| 3. Canlı performans | Modelin eğitim verisinden **sonra** oynanan 1. lig ve UEFA maçlarında, maç öncesi özelliklerle yaptığı tahminler puanlanır ve kapanış oranlarıyla karşılaştırılır. Sonuç `models/canli_performans.csv` dosyasına eklenir. | saniyeler |
| 4. Yeniden eğitim (koşullu) | Aşağıdaki tabloya bakın. | ~50 dk |
| 5. Simülasyon tablosu | `exports/sim_predictions_<tarih>.json` ve sabit adlı `exports/sim_predictions_latest.json` üretilir. | saniyeler |

**Neden her hafta yeniden eğitim yok?** Takımların güncel durumu özelliklerden okunur ve özellikler her hafta yenilenir. Modelin "özellik → olasılık" eşlemesi ise yavaş değişir; onu her hafta yeniden öğrenmek fayda sağlamaz.

**Yeniden eğitim ne zaman olur?** Aşağıdakilerden biri yeterlidir:

| Tetikleyici | Koşul |
|---|---|
| Zaman | Model 28 günden eski |
| Yeni sezon | Test sezonu değişti, yani bir sezon tamamlandı |
| Performans | En az 300 maçta modelin piyasa farkı, testteki farktan 0.02'den fazla kötü |

**Eğitim güvenli şekilde yapılır:**
- Yeni model önce `models/staging/` klasöründe eğitilir.
- **Kalite kapısı** yeni modelde üç şeyi kontrol eder:
  - Model yüklenip tahmin yapabiliyor mu.
  - CV ölçütü mevcut modelinkinden en fazla 0.01 kötü mü.
  - Test log-loss'u naif tahminden belirgin şekilde iyi mi.
- Kapı geçilirse mevcut model `models/arsiv/<zaman>/` klasörüne kopyalanır ve yeni model devreye alınır. Geçilemezse mevcut model kalır ve rapor uyarı verir.

**Takip:**
- Son çalıştırmanın raporu `reports/haftalik_rapor.md` dosyasında. Tüm geçmiş `reports/guncelleme_gecmisi.csv`, ayrıntılı loglar `logs/` klasöründe.
- Görev Zamanlayıcı'daki "Son çalıştırma sonucu" kodları: `0` = sorunsuz, `2` = tamamlandı ama uyarı var (rapora bakın), `1` = hata.
- Son maçı 400 günden eski olan takımlarda (veri kapsamı dışındaki ligler) Elo hesaplanamaz. Bu takımlar tabloya yazılmaz ve raporda listelenir. Tablonun geri kalanı yine güncellenir.

```bash
run_weekly.bat                    # elle çalıştır
run_weekly.bat --force-retrain    # yeniden eğitimi şimdi zorla
run_weekly.bat --no-retrain       # bu sefer yeniden eğitme
```

Görevi değiştirmek ya da kaldırmak için Görev Zamanlayıcı'yı açın (`taskschd.msc`) ya da PowerShell'de şu komutu kullanın:

```powershell
Unregister-ScheduledTask -TaskName "futbol-ml-modeli haftalik guncelleme"
```

Model yapısı değişirse (yeni özellik, yeni veri kaynağı) otomatik akış yeterli değildir. Bu durumda yeniden eğitimi elle başlatıp sonuçları inceleyin.

## Periyodik güncellenmesi gerekenler

Aşağıdakilerin hepsi, `team_aliases.csv` hariç, haftalık görev tarafından otomatik yapılır. Tablo, elle çalıştırmak gerektiğinde referans için burada duruyor.

| Ne | Ne sıklıkla | Neden |
|---|---|---|
| football-data | tahminden önce, sezon içinde haftalık | Elo, piyasa Elo'su, form, şut payı son maçlara bakar |
| Transfermarkt veri seti | yeni sürüm yayımlandıkça | UEFA maçları, ilk 11 ve hat değerleri. **Not:** veri setinin güncellemesi Temmuz 2026'da durdu. |
| `team_aliases.csv` | yeni takım / terfi / eşleşmeyen ad çıkınca | eşleşmeyen kulüplerin geçmişi birleşmez |
| Model (`python -m src.train`) | sezon sonunda; üretim için ardından `--refit-all` | yeni sezonu öğrenmek için |
| Simülasyon tahmin tablosu (`python -m src.export_predictions`) | veri güncellendikçe ya da simülasyonun takım listesi değişince | tablo, referans tarihteki takım durumunu yansıtır |

`predict.py`, özellik tablolarını `data/processed/feature_store.joblib` dosyasında önbelleğe alır. Ham veri değişince otomatik olarak yeniden kurulur; Elo parametreleri eğitimdeki değerler olarak kalır.

## Modeli başka bir projeye taşırken

**`feature_schema.md` tek doğruluk kaynağıdır.** Girdiler eğitimdekinden ufak da olsa farklı hesaplanırsa model hata vermez, sessizce yanlış olasılık üretir. Bu yüzden:

1. **İki dosyayı birlikte taşıyın.** `mac_modeli.pkl` ile `feature_schema.md` aynı sözleşme hash'ini taşır. Hash; özellik tanımlarını, Elo ve topluluk parametrelerini kapsar.
2. **Kütüphane versiyonları aynı olsun.** Şemadaki `xgboost`, `scikit-learn` ve `scipy` sürümlerini birebir kurun.
3. **Özellikleri şemadaki tanımla hesaplayın.** En güvenli yol `src/` klasörünü ve veri akışını olduğu gibi taşımak. Kendiniz yeniden yazarsanız, bilinen maçlar için `predictor.features_for(...)` çıktısıyla birebir karşılaştırın.
4. **Topluluk formülünü değiştirmeyin.** Referans uygulama `src/predict.py` dosyasındaki `component_outputs`, `combine` ve `consistent_score_matrix` fonksiyonları.
5. **NaN'ı doldurmayın.** Eksik değer işleme pkl'ın içindedir. `elo_diff` NaN ise tahmin yapmayın.
6. **Takım adları kanonik anahtarlara (`tm:<id>`) eşlenmeli.** Hedef projedeki adlar ("Bayern München", "Atleti" gibi) için bir eşleştirme tablosu hazırlayın.
7. **Çalışma anında da veri gerekir.** Güncel maç sonuçları, oranlar, şut istatistikleri ve Transfermarkt oyuncu verisi.
8. **`neutral`, `is_knockout` ve `is_uefa`'yı doğru verin.** Sınıf sırası `[home_win, draw, away_win]`.
9. **Doğrulama kontrollerini koruyun.** `assert` yerine `raise` kullanıldı, çünkü `python -O` assert'leri kapatır.

## Bilinen sınırlamalar

- **Transfermarkt veri seti 6 Temmuz 2026'da durdu.** 2025-26 UEFA finalleri ve 2026 yazı ön eleme maçları yok; hat değerleri yaz transferlerini yansıtmıyor.
- **İç ligi kapsanmayan ülkelerin kulüpleri** (Azerbaycan, Kıbrıs, Macaristan vb.) için Elo yalnızca UEFA maçlarından hesaplanıyor. Oyuncu ve şut özellikleri çoğunlukla NaN.
- **Şut verisi eksik.** 2017 öncesi pek çok ligde ve football-data "extra" liglerinde (İskandinavya, Polonya, Romanya, Avusturya, İsviçre) yok.
- **"Düzenli ilk 11" gerçek maç kadrosu değil.** Son 5 maçın dakikalarından tahmin ediliyor.
- **Transfermarkt piyasa değerleri** kitle kaynaklı tahminlerdir.
- **Sakatlık, ceza ve kesin kadro verisi yok.**

## Sonraki adımlar (etki sırasıyla)

1. **Sakatlık ve kesin kadro verisi (API-Football).** Piyasa ile aradaki farkın en büyük kaynağı. Kullanıcının API anahtarı gerekiyor; `load_injuries_and_suspensions` placeholder'ı hazır.
2. **Lisanslı xG verisi.** Şut payından daha güçlü oyun kalitesi göstergesi.
3. **Simülasyon entegrasyonu.** Simülasyon projesinde ayrı bir oturumda yapılacak; bkz. [ENTEGRASYON.md](ENTEGRASYON.md).
