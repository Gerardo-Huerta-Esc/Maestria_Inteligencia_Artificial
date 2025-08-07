# Segmentación Semántica con U-Net en PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Definir modelo U-Net básico
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))
        return torch.sigmoid(self.output(dec1))

# Cargar dataset de ejemplo (Pascal VOC)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

trainset = datasets.VOCSegmentation(root='.', year='2012', image_set='train', download=True,
                                    transform=transform, target_transform=target_transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

# Entrenamiento del modelo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, masks in trainloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# Visualización de resultados

def visualize(image, mask, pred):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.permute(1, 2, 0))
    axs[0].set_title("Imagen")
    axs[1].imshow(mask.squeeze(), cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred.squeeze(), cmap="gray")
    axs[2].set_title("Predicción")
    for ax in axs: ax.axis("off")
    plt.show()

model.eval()
with torch.no_grad():
    img, gt = next(iter(trainloader))
    img, gt = img.to(device), gt.to(device)
    pred = model(img)
    for i in range(2):
        visualize(img[i].cpu(), gt[i].cpu(), pred[i].cpu() > 0.5)  # umbral 0.5

# Guardar modelo
# torch.save(model.state_dict(), 'modelo_unet.pth')
