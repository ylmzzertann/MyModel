# Simülasyon entegrasyon rehberi

Bu dosya, modeli simülasyon projesine bağlayacak **ayrı oturum** için hazırlandı. Bu projede simülasyon koduna dokunulmadı. Simülasyondan yalnızca takım listesi dosyaları (`API/data/{ucl,uel,uecl}/*_teams.json`) okundu.

## Teslim edilenler

| Dosya | Ne işe yarar |
|---|---|
| `exports/sim_predictions_latest.json` (~0.9 MB) | **Simülasyonun okuması gereken dosya.** Her pazartesi otomatik yenilenir (bkz. README "Haftalık otomatik güncelleme"). 3 turnuvada tüm ev/deplasman eşleşmeleri × 3 maç türü. Tarihli kopyalar: `sim_predictions_<tarih>.json`. |
| `exports/sim_score_sampler.py` | Tablodan skor matrisi kurup skor örnekleyen referans kod. Yalnızca standart kütüphane kullanır; simülasyona yeni bağımlılık eklemez. |
| `data/sim_team_mapping.csv` | Simülasyon takım adı → model kulübü eşleştirmesi (99 ad; 15'i elle doğrulandı, kalanlar tek tek gözden geçirildi) |

Tablo, simülasyonun **kendi takım adlarıyla** anahtarlanır. Simülasyon tarafında ad eşleştirmesi gerekmez.

## Tablo formatı

```json
{
  "meta": {"reference_date": "2026-09-14", "rho": -0.0304, "max_goals": 10, "stages": {...},
           "score_matrix_recipe": [...], "model_trained_at_utc": "...", "contract_hash": "..."},
  "teams": {"ucl": [{"sim_name": "Paris", "key": "tm:583", "model_name": "Paris Saint-Germain", ...}]},
  "predictions": {
    "ucl": {"Galatasaray": {"Bayern München": {
        "league":   {"p": [0.186, 0.200, 0.614], "lambda": [1.10, 2.07]},
        "knockout": {"p": [...], "lambda": [...]},
        "final":    {"p": [...], "lambda": [...]}
    }}}
  }
}
```

- `p`: [ev sahibi kazanır, beraberlik, deplasman kazanır]. Bu, topluluk modelinin nihai olasılığıdır.
- `lambda`: [ev, deplasman] beklenen gol sayısı (90 dakika).
- **Maç türünü seç:**
  - `league`: lig aşaması maçları.
  - `knockout`: eleme turu maçları (play-off, son 16, çeyrek, yarı; iki ayaklı turlarda her ayak kendi ev sahibiyle).
  - `final`: tek maç, tarafsız saha.

## Skor üretimi

```python
from sim_score_sampler import load_table, get_entry, sample_score, sample_extra_time

TABLE = load_table("sim_predictions_latest.json")          # uygulama açılışında (ya da dosya değişince) bir kez
RHO = TABLE["meta"]["rho"]

entry = get_entry(TABLE, "ucl", home_name, away_name, "league")
home_goals, away_goals = sample_score(entry, RHO)           # 90 dakika
et_home, et_away = sample_extra_time(entry)                 # uzatma (yaklaşık: λ × 30/90)
```

`sample_score`, skor matrisinden örnekler. Bu matrisin ev kazanır / beraberlik / deplasman kazanır toplamları `p` ile birebir aynıdır. Yani çok sayıda simülasyonda sonuç dağılımı model olasılıklarını izler. Doğrulama: 20.000 örnekte %18,5 / %20,2 / %61,3; model %18,6 / %20,0 / %61,4. Penaltılar için simülasyonun mevcut mantığı kullanılabilir.

## İki entegrasyon seçeneği

1. **Tamamen değiştirmek.** Skor üretim fonksiyonundaki gol hesaplaması, tablodan `sample_score` ile değiştirilir. En tutarlı seçenek budur: sonuç olasılıkları doğrudan geçmiş veriyle kalibre edilmiş modelden gelir.
2. **Harmanlamak (kalibrasyon katmanı).** Mevcut formülün beklenen golleri modelinkilerle karıştırılır. Örneğin `λ = α·λ_model + (1-α)·λ_formül`, ardından mevcut Poisson çekimi yapılır.
   - Mevcut bonuslar (diziliş, kimya, yıldız oyuncu vb.) korunur.
   - Bu yolda `p` ile tam tutarlılık kaybolur. α=1 olduğunda bile Poisson çekimi Dixon-Coles ve `p` ölçeklemesini içermez.
   - α değeri, simülasyonda üretilen sonuç dağılımı tablodaki `p` ile karşılaştırılarak seçilebilir.

## Dikkat edilecekler

- **Tabloda olmayan takımlar:** Aşağıdaki durumlarda `get_entry` `KeyError` verir; mevcut formüle geri dönülmeli.
  - Kullanıcının eklediği özel kulüpler.
  - Takım listesi değişince eklenen takımlar.
  - Son 400 günde veri setinde maçı olmayan, bu yüzden Elo'su hesaplanamayan takımlar. Bunlar `meta.teams_without_prediction` listesinde görünür; 21.09.2026 itibarıyla yalnızca `ucl:Sabah`.
- **Dosya güncellemesi:** `sim_predictions_latest.json` her hafta atomik olarak (önce geçici dosyaya yazılıp) değiştirilir; yarım yazılmış dosya okunmaz. Simülasyon, dosyanın değiştirilme zamanına bakarak tabloyu yeniden yükleyebilir.
- **Takım listesi değişince:** bu projede tabloyu yeniden üret:
  ```bash
  python -m src.export_predictions --teams-dir "<simülasyon>/API/data" --date YYYY-MM-DD
  ```
  Dosyadaki doğrulanmış eşleştirmeler korunur. Yeni takımlar otomatik eşlenir ve gözden geçirilmelidir. Eşlenemeyen takım olursa tablo yazılmaz.
- **Referans tarih:** tahminler, bu tarihten önceki verilerle hesaplanmış takım durumunu yansıtır. Sezon içinde güncellemek için veriyi indirip tabloyu yeni tarihle yeniden üret.
- **Eksik bilgi:** 99 takımın 40'ında oyuncu verisi, 48'inde şut verisi yok. Bunlar Avusturya, İsviçre, Polonya, Romanya, Çekya, İskandinavya ve küçük ülke kulüpleri. Bu takımların tahminleri ağırlıklı olarak Elo ve piyasa Elo'suna dayanır.
- **Veri güncelliği:** Transfermarkt verisi Temmuz 2026'da durdu; oyuncu değerleri yaz transferlerini içermiyor.
- **Ölçek farkı:** simülasyonun FC27 oyuncu reytingleri, modelin kullandığı Transfermarkt piyasa değerlerinden farklı bir ölçekte. Bu reytingler modele girdi olarak verilemez.
