# Haftalık güncelleme raporu — 2026-09-28 13:39

**Durum:** TAMAMLANDI (uyarılarla)

## Veri

- football-data: 333 dosya hazır
- transfermarkt: 6 dosya hazır
- Kaynakların son maç tarihleri: {'football-data': '2026-09-27', 'transfermarkt': '2026-05-24'}

## Model

- Eğitim zamanı: 2026-09-21 06:47:02
- Eğitim verisinin son maçı: 2026-09-17

## Canlı performans (eğitimden sonra oynanan maçlar)

- Dönem: 2026-09-17 sonrası → 2026-09-25, 160 maç
- Log-loss 1.0217, Brier 0.6120, accuracy 0.500
- Kapanış oranlarıyla kıyas (160 maç): model 1.0217, piyasa 0.9793, fark +0.0424
- Not: 300 maçtan az; değerler henüz gürültülü.

## Yeniden eğitim

- Gerekçe: gerek yok
- Sonuç: yapılmadı

## Simülasyon tablosu

- sim_predictions_2026-09-28.json + sim_predictions_latest.json (11130 tahmin, referans tarih 2026-09-28)

## Uyarılar

- Elo'su olmayan (son 400 günde maçı olmayan) takımlar tabloya yazılmadı, simülasyon bunlar için kendi formülünü kullanır: ['ucl:Sabah']
