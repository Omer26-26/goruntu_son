# Görüntü İşleme Projesi

Python ile hazırlanmış masaüstü görüntü işleme uygulamasıdır. İşlemler hazır görüntü işleme fonksiyonları kullanılmadan, NumPy ve manuel algoritma mantığıyla yazılmıştır.

## Çalıştırma

```bash
python main.py
```

## Gerekli Kütüphaneler

```bash
pip install numpy pillow matplotlib customtkinter
```

## Uygulamadaki İşlemler

- Gri seviye ve thresholding
- Kontrast, histogram germe ve histogram eşitleme
- RGB-HSV ve RGB-YCbCr renk dönüşümleri
- Görüntü döndürme, zoom ve kırpma
- Salt & Pepper gürültüsü ekleme
- Mean, Median ve Motion Blur filtreleri
- Sobel ve Canny kenar bulma
- Erosion, Dilation, Opening ve Closing morfolojik işlemleri
- İki görüntü ile aritmetik işlemler: toplama, çıkarma, çarpma, AND, OR, XOR
- Adaptif eşikleme ve çift eşikleme

## Dosyalar

- `main.py`: Arayüz ve işlem seçimleri
- `Processor.py`: Görüntü işleme algoritmaları
- `Image_Capture.py`: Görüntü okuma/gösterme yardımcıları
- `test.py`: Basit test dosyası
