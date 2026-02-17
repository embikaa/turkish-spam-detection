#!/bin/bash
# Kurulum ve Kullanım Scripti

echo "=================================="
echo "Turkish Spam Detection - Kurulum"
echo "=================================="

# 1. Virtual environment oluştur (sadece ilk kez)
if [ ! -d "venv" ]; then
    echo "Virtual environment oluşturuluyor..."
    python3 -m venv venv
fi

# 2. Virtual environment'ı aktive et
echo "Virtual environment aktive ediliyor..."
source venv/bin/activate

# 3. Bağımlılıkları yükle
echo "Bağımlılıklar yükleniyor (bu işlem birkaç dakika sürebilir)..."
pip install -r requirements.txt

echo ""
echo "✅ Kurulum tamamlandı!"
echo ""
echo "Kullanım:"
echo "  1. Virtual environment'ı aktive edin: source venv/bin/activate"
echo "  2. Modeli eğitin: python train.py"
echo "  3. Tahmin yapın: python predict.py"
echo "  4. API başlatın: python api.py"
echo ""
