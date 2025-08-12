from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import os

app = Flask(__name__)

# Model dosya yolunu dinamik olarak ayarla
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/model.onnx')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"ONNX model dosyası bulunamadı: {MODEL_PATH}\n"
        "Lütfen önce eğitim scriptini çalıştırarak modeli oluşturun:\n"
        "    python ../src/train.py"
    )

# ONNX session
sess = ort.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name

transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

LABELS = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'file missing'}), 400
    f = request.files['file']
    img = Image.open(f.stream).convert('RGB')
    img = transform(img).unsqueeze(0).numpy().astype(np.float32)

    preds = sess.run(None, {input_name: img})[0]
    probs = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)
    top_idx = int(np.argmax(probs, axis=1)[0])
    top_prob = float(probs[0, top_idx])
    return jsonify({'class': LABELS[top_idx], 'probability': top_prob})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)