import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Burada çok basit bir sinir ağı (CNN) modeli tanımlıyoruz.
# CNN, resimlerdeki özellikleri otomatik olarak bulabilen bir yapay zeka modelidir.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.25)
        # 128 kanal, 4x4 çıktı boyutu (her havuzlamadan sonra boyut yarıya iner, 28->26->24->22->20, 2x2 pooling ile 10->5, padding yoksa 4x4 olabilir)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy, np.array(all_preds).flatten(), np.array(all_targets).flatten()

def train_loop(model, device, train_loader, val_loader, optimizer, epochs=10, patience=3, scheduler=None, writer=None):
    criterion = nn.NLLLoss()
    best_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    lrs = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        val_loss, val_acc, _, _ = validate(model, device, val_loader, criterion)
        print(f"Epoch {epoch}: Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        lrs.append(optimizer.param_groups[0]['lr'])

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model kaydedildi. Epoch {epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping tetiklendi.")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses, train_accuracies, val_accuracies, lrs

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomCrop(28, padding=4),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Eğitim ve validation için veri setini böl
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    writer = SummaryWriter(log_dir='runs/mnist_experiment')

    train_losses, val_losses, train_accuracies, val_accuracies, lrs = train_loop(
        model, device, train_loader, val_loader, optimizer, epochs=5, patience=3, scheduler=scheduler, writer=writer
    )

    # Sadece ağırlıkları değil, tüm modeli kaydet
    torch.save(model, 'mnist_cnn_fullmodel.pth')
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model ve ağırlıklar kaydedildi: mnist_cnn_fullmodel.pth, mnist_cnn.pth")

    # Son epoch için confusion matrix
    criterion = nn.NLLLoss()
    _, _, preds, targets = validate(model, device, val_loader, criterion)
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.show()

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    writer.close()

    # Eğitilmiş modeli yükleyip tek bir resim üzerinde test et
    loaded_model = torch.load('mnist_cnn_fullmodel.pth', map_location=device)
    loaded_model.eval()

    # Validation setinden bir örnek al
    sample_img, sample_label = val_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = loaded_model(sample_img)
        pred_label = output.argmax(dim=1, keepdim=True)

    # Sonucu görselleştir
    plt.figure(figsize=(8,8))
    plt.imshow(sample_img.cpu().squeeze(0), cmap='gray')
    plt.title(f"Gerçek Etiket: {sample_label}, Tahmin Edilen Etiket: {pred_label.item()}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
    print("Egitim tamamlandi.")  # Eğitim bittiğinde ekrana yaz