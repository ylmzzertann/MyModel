# Haftalık güncelleme raporu — 2026-10-19 10:00

**Durum:** TAMAMLANDI (uyarılarla)

## Veri

- football-data: 333 dosya hazır
- transfermarkt: 6 dosya hazır
- Kaynakların son maç tarihleri: {'football-data': '2026-10-14', 'transfermarkt': '2026-05-24'}

## Model

- Eğitim zamanı: 2026-10-19 14:10:05
- Eğitim verisinin son maçı: 2026-10-12

## Canlı performans (eğitimden sonra oynanan maçlar)

- Dönem: 2026-09-17 sonrası → 2026-10-12, 322 maç
- Log-loss 0.9706, Brier 0.5774, accuracy 0.562
- Kapanış oranlarıyla kıyas (322 maç): model 0.9706, piyasa 0.9461, fark +0.0245

## Yeniden eğitim

- Gerekçe: model 28 gün önce eğitildi (sınır 28)
- Sonuç: YENİ MODEL DEVREDE (önceki model arşivde: models/arsiv/2026-10-19_1010)
  - ✔ yeni model yüklenip tahmin yapabiliyor
  - ✔ CV ölçütü 0.95893 <= mevcut 0.95893 + 0.01
  - ✔ test log-loss 0.9954 < naif 1.0713 - 0.03

## Simülasyon tablosu

- sim_predictions_2026-10-19.json + sim_predictions_latest.json (11130 tahmin, referans tarih 2026-10-19)

## Uyarılar

- Elo'su olmayan (son 400 günde maçı olmayan) takımlar tabloya yazılmadı, simülasyon bunlar için kendi formülünü kullanır: ['ucl:Sabah']
