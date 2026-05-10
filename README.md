# Görüntü İşleme Ödevi

Bu proje, Numpy kütüphanesi kullanılarak temel görüntü işleme algoritmalarının manuel olarak (OpenCV hazır fonksiyonları kullanılmadan) gerçeklendiği bir masaüstü uygulamasıdır.

## Özellikler

- **Gri ve Binary Dönüşüm**: Görüntüyü gri tonlamalı ve siyah-beyaz formatına çevirir.
- **Döndürme ve Kırpma**: Görüntüyü 90 derece döndürür ve belirli alanları maskeler (kırpar).
- **Filtreler**: Median, Mean, Gaussian ve Motion Blur filtreleri.
- **Kenar Bulma**: Sobel algoritması ile kenar tespiti.
- **Morfolojik İşlemler**: Aşınma, Genişleme, Açma ve Kapama.
- **Histogram**: Histogram hesaplama ve germe işlemleri.
- **Aritmetik İşlemler**: İki görüntü arasında toplama, çıkarma ve çarpma.

## Kullanım

1. `main.py` dosyasını çalıştırın.
2. Sol menüden "Resim Yükle" butonuna basarak bir görsel seçin.
3. Filtre listesinden istediğiniz işlemi seçip "Uygula" butonuna basın.
4. "Görüntü Döndürme" her uygulandığında resmi 90 derece döndürecektir.

## Gereksinimler

- Python 3.x
- numpy
- customtkinter
- Pillow
- matplotlib
- opencv-python (Görüntü okuma/yazma ve test amaçlı)
