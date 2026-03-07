from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size=28*28, h1=32, output_size=27):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, output_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # 32 -> 16 -> 8 -> 4
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.flatten(1)
        return self.fc(x)

class SmallConvBigHead(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        # feature extractor molto piccolo
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1x28x28 -> 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # -> 8x14x14
        )

        # testa fully connected più grande
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MicroConvTinyHead(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        # Aggiungiamo un pooling in più per abbattere le dimensioni spaziali
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1x28x28 -> 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 8x14x14
            nn.MaxPool2d(2)  # -> 8x7x7 (strozzatura spaziale!)
        )

        # Testa molto più piccola e con un layer in meno
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 32),  # 392 -> 32 (Invece di 1568 -> 256)
            nn.ReLU(),
            nn.Linear(32, num_classes)  # 32 -> 27 (Rimosso il layer da 128)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x