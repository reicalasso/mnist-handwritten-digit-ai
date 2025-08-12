import torch
from model import ResNet18_1ch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model yükle
model = ResNet18_1ch(num_classes=10)
state = torch.load('../models/best_model.pth', map_location=DEVICE)
model.load_state_dict(state)
model.eval().to(DEVICE)

# TorchScript ile kaydetme (trace)
dummy = torch.randn(1, 1, 28, 28).to(DEVICE)
traced = torch.jit.trace(model, dummy)
traced.save('../models/model_scripted.pt')
print('TorchScript kaydedildi: ../models/model_scripted.pt')

# ONNX export
onnx_path = '../models/model.onnx'
input_names = ['input']
output_names = ['output']
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f'ONNX kaydedildi: {onnx_path}')