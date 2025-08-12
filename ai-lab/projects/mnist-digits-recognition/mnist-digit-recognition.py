import torch
from torchvision import datasets, transforms

#MNIST'i Tensor formatına dönüştürmek için transform kullandık.
transform = transforms.ToTensor()

#Egitim ve test setlerini indiriyoruz boring ass
train_dataset = datasets.MNIST(root='./data' , train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data' , train=False, transform=transform, download=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

#ilk veriyi ve etiketini inceleyenzi
image, label = train_dataset[0]
print(f"Image shape: {image.shape}") # (1, 28, 28) tensor boyutu
print(f"Label: {label}")