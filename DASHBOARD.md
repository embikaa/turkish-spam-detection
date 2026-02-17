# 🎨 Web Dashboard Kullanım Kılavuzu

## 🚀 Dashboard'u Başlatma

### 1. Virtual Environment'ı Aktive Edin
```bash
source venv/bin/activate
```

### 2. Dashboard'u Başlatın
```bash
python app.py
```

Çıktı:
```
============================================================
🚀 Turkish Spam Detection Dashboard
============================================================
📊 Dashboard: http://localhost:8000
📖 API Docs: http://localhost:8000/docs
============================================================
```

### 3. Tarayıcıda Açın
```
http://localhost:8000
```

---

## 🎯 Dashboard Özellikleri

### 1. **Model Bilgileri (Üst Banner)**
- Model versiyonu (timestamp)
- Accuracy, F1 Score, Precision, Recall değerleri
- Otomatik olarak yüklenir

### 2. **Yorum Testi (Sol Panel)**
- Kendi yorumunuzu yazın
- "Analiz Et" butonuna tıklayın
- Gerçek zamanlı sonuç:
  - 🟢 GERÇEK veya 🔴 SPAM
  - Spam ihtimali (%)
  - Güven seviyesi (Yüksek/Orta/Düşük)

**Örnek Yorumlar:**
```
✅ Gerçek: "Harika ürün, çok memnun kaldım. Detaylı inceleme yaptım ve kalitesi gerçekten çok iyi. Fiyat performans açısından da mükemmel."

❌ Spam: "süper"
❌ Spam: "ÇOOK İYİİİ!!!"
❌ Spam: "Harika ürün 👍👍👍"
```

### 3. **Model Performansı (Sağ Panel)**
- **Metrik Kartları**: Accuracy, F1, Precision, Recall
- **Bar Chart**: Tüm metriklerin görsel karşılaştırması

### 4. **Confusion Matrix (Sol Alt)**
- True Negative (Doğru Genuine)
- False Positive (Yanlış Spam)
- False Negative (Kaçan Spam)
- True Positive (Doğru Spam)
- Bar chart formatında

### 5. **ROC Curve (Sağ Alt)**
- Receiver Operating Characteristic eğrisi
- AUC (Area Under Curve) değeri
- Random classifier karşılaştırması

### 6. **Sınıf Dağılımı (En Alt)**
- Genuine vs Spam oranları
- Donut chart formatında
- Test setindeki dağılım

---

## 🎨 Tasarım Özellikleri

### Modern Dark Theme
- Gradient renkler (Mor-Mavi tonları)
- Glassmorphism efektleri
- Smooth animasyonlar

### Responsive Design
- Desktop, tablet, mobil uyumlu
- Grid layout otomatik ayarlanır

### Interactive Charts
- Chart.js ile dinamik grafikler
- Hover efektleri
- Animasyonlu geçişler

---

## 🔧 API Endpoints

Dashboard arka planda şu endpoint'leri kullanır:

### `GET /`
Ana dashboard sayfası

### `GET /model-info`
Model bilgilerini döner:
```json
{
  "version": "20260215_214656",
  "metrics": {
    "accuracy": 0.9220,
    "f1_score": 0.8561,
    "precision": 0.9206,
    "recall": 0.8000,
    "confusion_matrix": [[TN, FP], [FN, TP]],
    "class_distribution": {"Genuine": 710, "Spam": 290}
  }
}
```

### `POST /predict`
Spam tahmini yapar:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Harika ürün!"}'
```

### `GET /health`
Sistem sağlık kontrolü

### `GET /docs`
Swagger UI - Otomatik API dokümantasyonu

---

## 📸 Ekran Görüntüleri İçin

Dashboard'u çalıştırıp şunları yapın:

1. **Ana görünüm**: Tüm panellerin göründüğü ekran
2. **Gerçek yorum testi**: Yeşil sonuç
3. **Spam yorum testi**: Kırmızı sonuç
4. **Metrik grafikleri**: Yakınlaştırılmış görünüm

Tezinizde kullanabilirsiniz!

---

## 🎓 Tez İçin Açıklama

### Sistem Mimarisi
```
Kullanıcı → Web Arayüzü (HTML/CSS/JS)
              ↓
         FastAPI Backend
              ↓
    SpamDetectionPipeline
              ↓
    BERTurk + TF-IDF + RF
              ↓
         Tahmin Sonucu
```

### Teknolojiler
- **Frontend**: HTML5, CSS3 (Custom), Vanilla JavaScript
- **Charts**: Chart.js 4.4.0
- **Backend**: FastAPI 0.129.0
- **ML Pipeline**: PyTorch, Transformers, Scikit-learn

### Özellikler
- Real-time prediction
- Interactive visualizations
- Responsive design
- RESTful API
- Model versioning
- Comprehensive metrics

---

## 🐛 Sorun Giderme

### Dashboard açılmıyor
```bash
# Model eğitilmiş mi kontrol edin
ls models/latest/

# Eğer yoksa:
python train.py
```

### Grafikler görünmüyor
- Tarayıcı konsolunu açın (F12)
- JavaScript hataları var mı kontrol edin
- İnternet bağlantısı var mı? (Chart.js CDN için)

### API hatası
```bash
# Logları kontrol edin
tail -f logs/training.log
```

---

## 🚀 Production Deployment

### Lokal Ağda Paylaşım
```bash
# Tüm IP'lerden erişim
python app.py
# http://YOUR_IP:8000
```

### Sunucuya Deploy
```bash
# Gunicorn ile (production)
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.14
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -r requirements-api.txt
CMD ["python", "app.py"]
```

---

## 💡 İpuçları

1. **Farklı yorumlar deneyin**: Kısa, uzun, emoji'li, ALL CAPS
2. **Metrikleri not edin**: Teziniz için
3. **Ekran görüntüleri alın**: Görsel zenginlik
4. **API'yi test edin**: `/docs` sayfasından
5. **Farklı modeller karşılaştırın**: Yeniden eğitip metrik değişimini görün

Başarılar! 🎓
