import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from model import FashionCNN

# Hiperparametreler
BATCH_SIZE = 32  # Daha küçük batch size
LR = 5e-4        # Daha düşük learning rate
WEIGHT_DECAY = 5e-4
EPOCHS = 50      # Çok daha uzun eğitim
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloaders(root='./data'):
    # Eğitim için güçlü augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
    ])
    
    # Test/validation için sadece normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Eğitim ve test setlerini ayrı ayrı yükle
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_transform)
    
    # Validation için train setinden ayır
    n = len(train_dataset)
    val_size = int(0.05 * n)  # Sadece %5 validation
    train_size = n - val_size
    
    # İndeksler ile ayır
    indices = torch.randperm(n).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_set = Subset(train_dataset, train_indices)
    # Validation için test transform kullan
    val_dataset = datasets.FashionMNIST(root=root, train=True, download=False, transform=test_transform)
    val_set = Subset(val_dataset, val_indices)

    # Pin memory sadece CUDA varsa kullan
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    return train_loader, val_loader, test_loader

def train():
    train_loader, val_loader, test_loader = get_dataloaders()
    model = FashionCNN(num_classes=10).to(DEVICE)
    
    # Loss function ve optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # Warmup ve cosine annealing
    warmup_epochs = 5
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_epochs])

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    os.makedirs('../models', exist_ok=True)

    # Gradient scaler'ı loop dışında tanımla
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            # Mixed precision training sadece CUDA varsa
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            loop.set_postfix(loss=running_loss/total, acc=correct/total, lr=optimizer.param_groups[0]['lr'])

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                v_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        
        scheduler.step()

        print(f"Epoch {epoch}: Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # En iyi model kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../models/best_model.pth')
            print(f"🎯 Yeni en iyi model kaydedildi: val_acc={val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️  Early stopping tetiklendi.")
                break

    # Eğitim bittiğinde test set performansı
    model.load_state_dict(torch.load('../models/best_model.pth'))
    model.to(DEVICE).eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += imgs.size(0)
    print(f"Test loss: {test_loss/test_total:.4f}, Test acc: {test_correct/test_total:.4f}")

    # TorchScript ve ONNX olarak kaydet
    dummy_input = torch.randn(1, 1, 28, 28, device=DEVICE)
    ts_path = '../models/model_ts.pt'
    onnx_path = '../models/model.onnx'
    # TorchScript
    traced = torch.jit.trace(model, dummy_input)
    traced.save(ts_path)
    print(f"TorchScript model kaydedildi: {ts_path}")
    # ONNX
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    print(f"ONNX model kaydedildi: {onnx_path}")

if __name__ == '__main__':
    train()