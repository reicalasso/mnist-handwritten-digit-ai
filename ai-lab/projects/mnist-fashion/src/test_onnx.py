import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_set = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
loader = DataLoader(test_set, batch_size=256, shuffle=False)

sess = ort.InferenceSession('../models/model.onnx')
input_name = sess.get_inputs()[0].name

correct = 0
total = 0
for imgs, labels in loader:
    # imgs: torch tensor -> numpy
    x = imgs.numpy().astype(np.float32)
    # ONNX Runtime expects shape (N, C, H, W)
    preds = sess.run(None, {input_name: x})[0]
    preds_labels = np.argmax(preds, axis=1)
    correct += (preds_labels == labels.numpy()).sum()
    total += labels.size(0)

print(f'ONNX test accuracy: {correct/total:.4f} ({correct}/{total})')