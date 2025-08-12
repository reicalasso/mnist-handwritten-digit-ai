import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensorflow.keras.callbacks import EarlyStopping

# Burada çok basit bir sinir ağı (CNN) modeli tanımlıyoruz.
# CNN, resimlerdeki özellikleri otomatik olarak bulabilen bir yapay zeka modelidir.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # İlk katman: 1 kanallı (gri tonlu) resmi alır, 32 farklı filtre uygular, her filtre 3x3 boyutunda
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # 1 giriş kanalı, 32 çıkış kanalı, 3x3 filtre, 1 adımda kay
        # İkinci katman: 32 kanallı çıktıyı alır, 64 filtre uygular
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout: Aşırı öğrenmeyi engellemek için bazı nöronları rastgele kapatır (0.25 oranında)
        self.dropout1 = nn.Dropout2d(0.25)
        # Tam bağlantılı katman: 64 kanallı, 12x12 boyutunda veriyi (toplam 9216 değer) 128 nörona bağlar
        self.fc1 = nn.Linear(9216, 128)
        # Dropout: Yine aşırı öğrenmeyi engellemek için (0.5 oranında)
        self.dropout2 = nn.Dropout(0.5)
        # Son katman: 128 nörondan 10 çıkış (0'dan 9'a kadar rakamlar için)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # İleri besleme işlemi: Veriyi katmanlardan geçiriyoruz
        x = self.conv1(x)  # İlk konvolüsyon katmanı
        x = torch.relu(x)  # Aktivasyon fonksiyonu: Negatifleri sıfırlar, pozitifleri bırakır
        x = self.conv2(x)  # İkinci konvolüsyon katmanı
        x = torch.relu(x)  # Yine aktivasyon
        x = torch.max_pool2d(x, 2)  # 2x2 boyutunda havuzlama: Veriyi küçültür, önemli bilgileri tutar
        x = self.dropout1(x)  # Dropout uygula
        x = torch.flatten(x, 1)  # Çok boyutlu veriyi tek boyuta indir (batch boyutu hariç)
        x = self.fc1(x)  # Tam bağlantılı katman
        x = torch.relu(x)  # Aktivasyon
        x = self.dropout2(x)  # Dropout uygula
        x = self.fc2(x)  # Son katman
        output = torch.log_softmax(x, dim=1)  # Sonuçları olasılığa çevir (logaritmik olarak)
        return output

# Bu fonksiyon modeli eğitir. Yani modele verileri gösterip doğruyu öğrenmesini sağlar.
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # Modeli eğitim moduna al
    criterion = nn.NLLLoss()  # Kayıp fonksiyonu: Tahmin ile gerçek arasındaki farkı ölçer
    for batch_idx, (data, target) in enumerate(train_loader):
        # data: Resimler, target: Doğru rakamlar
        data, target = data.to(device), target.to(device)  # Veriyi cihaza (CPU/GPU) gönder

        optimizer.zero_grad()  # Önceki gradyanları sıfırla
        output = model(data)  # Modelden tahmin al
        loss = criterion(output, target)  # Tahmin ile gerçek arasındaki farkı hesapla
        loss.backward()  # Geri yayılım: Hataları geriye doğru dağıt
        optimizer.step()  # Modelin ağırlıklarını güncelle

        # Her 100 adımda bir ekrana kayıp değerini yazdır
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# Erken durdurma için fonksiyon
def early_stop(model, device, train_loader, optimizer, epoch, patience=3):
    # Eğitim sırasında kaybın iyileşmediği epoch'ları izler
    # Eğer belirli bir sabır (patience) sayısı kadar epoch boyunca iyileşme olmazsa, eğitimi durdurur
    model.eval()  # Modeli değerlendirme moduna al
    criterion = nn.NLLLoss()
    # İlk kaybı hesapla
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    min_loss = loss.item()
    
    # Sabır sayısını başlat
    trigger_times = 0 

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        # Eğer kayıp önceki en düşük kayıptan daha düşükse, modeli ve kaybı güncelle
        if loss.item() < min_loss:
            min_loss = loss.item()
            trigger_times = 0 
            # Modelin ağırlıklarını kaydet
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model kaydedildi: best_model.pth (Epoch: {epoch}, Batch: {batch_idx})")
        else:
            trigger_times += 1 

        # Eğer sabır sayısı aşılırsa, eğitimi durdur
        if trigger_times >= patience:
            print(f"Erken durdurma: Kayıp {patience} dönemdir iyileşmiyor.")
            return True 
    return False 

def main():
    # Cihaz ayarlama: Eğer bilgisayarda ekran kartı (GPU) varsa onu kullan, yoksa işlemci (CPU) kullan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST veri setini indir ve uygun şekilde dönüştür
    # transforms.ToTensor(): Resmi tensöre çevirir (PyTorch'un anlayacağı formata)
    # transforms.Normalize: Veriyi ortalaması 0, standart sapması 1 olacak şekilde normalleştirir
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Resimleri rastgele döndür (10 dereceye kadar)
        transforms.RandomHorizontalFlip(),  # Resimleri rastgele yatay çevir/MNIST icin biraz tartismali
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Eğitim verisini indir ve yükle
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 64'lük gruplar halinde karıştırarak yükle

    # Modeli oluştur ve cihaza gönder
    model = SimpleCNN().to(device)
    # Adam optimizasyon algoritmasını kullan, öğrenme oranı 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Ağırlıkların aşırı öğrenmesini engellemek için ağırlık çürümesi (weight decay) ekle

    # Eğitim döngüsü: Modeli 10 kez (epoch) tüm verilerle eğit
    epochs = 10  # 10 kere tüm veriyi göster
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        # Erken durdurma kontrolü
        if early_stop(model, device, train_loader, optimizer, epoch):
            break 

# Modelin ağırlıklarını dosyaya kaydet (sonradan tekrar kullanmak için)
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model kaydedildi: mnist_cnn.pth")

if __name__ == '__main__':
    main()
    print("Egitim tamamlandi.")  # Eğitim bittiğinde ekrana yaz