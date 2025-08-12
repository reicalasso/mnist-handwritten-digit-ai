# FashionMNIST CNN + ONNX + Flask API

Bu proje, PyTorch ile eğitilmiş bir CNN modelini kullanarak FashionMNIST veri setindeki görüntüleri sınıflandırır. Model hem **TorchScript** hem de **ONNX** formatlarında kaydedilir ve Flask tabanlı bir REST API üzerinden tahmin yapılabilir. Ayrıca HTML/JavaScript arayüzü ile görsel olarak test imkanı sunar.

---

## 🚀 Özellikler
- **PyTorch** ile CNN eğitimi
- **ONNX Runtime** ile hızlı inference
- **Flask REST API** üzerinden tahmin
- **HTML/JS Arayüz** ile kolay kullanım
- Modeli TorchScript ve ONNX olarak kaydetme

---

## Kurulum ve Kullanım

### 1. Ortamı Hazırlama

Öncelikle Python ortamını kurun:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` içinde PyTorch, torchvision, Flask, ONNX Runtime vb. kütüphaneler tanımlı olacak.

### 2. Modeli Eğitme

Modelinizi FashionMNIST ile eğitmek için:

```bash
python train.py
```

Bu script, modeli eğitir.

Eğitim bitince iki formatta kayıt yapılır:

- TorchScript → `model_ts.pt`
- ONNX → `model.onnx`

### 3. ONNX Runtime ile Test

```bash
python test_onnx.py
```

`test_onnx.py`, kaydedilmiş `model.onnx` dosyasını ONNX Runtime ile açar.

Test veri seti üzerinde doğruluk (accuracy) verir.

### 4. REST API’yi Başlatma

```bash
python app.py
```

Flask API `http://127.0.0.1:5000` üzerinde çalışır.

`/predict` endpoint’i görsel dosyayı alır, modeli kullanarak sınıflandırma yapar.

### 5. HTML/JS Arayüzünü Açma

`static/index.html` dosyasını tarayıcıda açın.

Bir resim seçip “Tahmin Et” düğmesine basarak API’ye istekte bulunun.

Tahmin sonucu ekranda görünür.
