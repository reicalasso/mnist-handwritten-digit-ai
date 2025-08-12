import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Burada çok basit bir sinir ağı (CNN) modeli tanımlıyoruz.
# CNN, resimlerdeki özellikleri otomatik olarak bulabilen bir yapay zeka modelidir.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)      # Yeni katman
        self.conv4 = nn.Conv2d(128, 128, 3, 1)     # Yeni katman
        self.dropout1 = nn.Dropout2d(0.25)
        # 128 kanal, 4x4 çıktı boyutu (her havuzlamadan sonra boyut yarıya iner, 28->26->24->22->20, 2x2 pooling ile 10->5, padding yoksa 4x4 olabilir)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.conv3(x)         # Yeni katman
        x = torch.relu(x)
        x = self.conv4(x)         # Yeni katman
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy

def train_loop(model, device, train_loader, val_loader, optimizer, epochs=10, patience=3):
    criterion = nn.NLLLoss()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print(f"Epoch {epoch}: Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

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

    # En iyi ağırlıkları yükle
    model.load_state_dict(torch.load('best_model.pth'))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), # RandomAffine eklendi
        # RandomAffine, resimlerin rastgele döndürülmesi, kaydırılması,
        # ölçeklenmesi ve eğilmesi gibi dönüşümler uygular.
        # Bu, modelin daha çeşitli verilerle eğitilmesini sağlar.
        # Bu dönüşümler, modelin daha iyi genelleme yapabilmesini sağlar.
        # Örneğin, el yazısı rakamların farklı açılarda ve boyutlarda yazılmasını simüle eder.
        # Bu, modelin daha iyi genelleme yapabilmesini sağlar.
        # RandomCrop, resimlerin rastgele kesilmesini sağlar.
        # Bu, modelin daha iyi genelleme yapabilmesini sağlar.
        # GaussianBlur, resimlerin bulanıklaştırılmasını sağlar.
        transforms.RandomCrop(28, padding=4), # RandomCrop eklendi
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # GaussianBlur eklendi
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_loop(model, device, train_loader, val_loader, optimizer, epochs=10, patience=3)

    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model kaydedildi: mnist_cnn.pth")

if __name__ == '__main__':
    main()
    print("Egitim tamamlandi.")  # Eğitim bittiğinde ekrana yaz