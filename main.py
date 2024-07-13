import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim #import optim
from PIL import Image
import pickle
import matplotlib.pyplot as plt

def default_conv(in_channels, out_channels, kernal_size, bias=True):            # 3 , 64 , 3
    return nn.Conv2d(in_channels, out_channels, kernal_size, padding=(kernal_size//2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        # Convert to numpy and plot each channel
        y_np = y.detach().cpu().numpy()[0]  # Assuming batch size of 1
        num_channels = y_np.shape[0]
        fig, axes = plt.subplots(1, num_channels, figsize=(20, 5))
        for i in range(num_channels):
            ax = axes[i]
            ax.imshow(y_np[i], cmap='gray')  # Removed unnecessary indices
            ax.axis('off')
        plt.show()
        return x * y

class Block1(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block1, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))  # Apply conv1 and then ReLU
        res = res + x  # Residual connection
        # Print and visualize the output of the convolutional layer
        num_channels = res.size(1)
        fig, axes = plt.subplots(1, num_channels, figsize=(20, 5))
        for i in range(num_channels):
            ax = axes[i]
            ax.imshow(res[0, i].detach().cpu().numpy(), cmap='gray')  # Assuming batch size of 1
            ax.axis('off')
        plt.show()
        res = self.conv2(res)  # Apply conv2
        res = self.calayer(res)  # Apply CALayer
        res = self.palayer(res)  # Apply PALayer
        res = res + x  # Another residual connection
        return res

class Block(nn.Module):
    def __init__(self ,conv, dim, kernal_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernal_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernal_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res = res + x
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernal_size, blocks):
        super(Group, self).__init__()
        modules = [Block1(conv, dim, kernal_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernal_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res = res + x
        return res

class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernal_size = 3
        pre_process = [conv(3, self.dim, kernal_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernal_size=kernal_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernal_size=kernal_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernal_size=kernal_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim*self.gps, 1, padding=1, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernal_size),
            conv(self.dim, 3, kernal_size)
        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        #print(x1.shape)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        w = self.ca(torch.cat([res1, res2, res3], dim = 1))
        #w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        w = w.view(-1, self.gps, self.dim)
        #print("Working Till Here")
        #print(w[:, 0, ::].shape)
        #print(res1.shape)
        out = (w[:, 0, ::] * res1) + (w[:, 1, ::] * res2) + (w[:, 2, ::] * res3)
        out = self.palayer(out)
        x = self.palayer(out)
        x = self.post(out)
        return x + x1

def load_image(Image_path, transform=None):
    image = Image.open(Image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image.unsqueeze(0)       # Adding Batch dimension

learning_rate = 1e-4
transform = transforms.Compose([
    transforms.ToTensor()
])

#Checking for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loader
low = '/hazy/image.png'
high = '/clear/image.png'

# Model, loss, optimizer
model = FFA(gps=3, blocks=19).to(device)
criterion = nn.MSELoss()            # L2 loss for image dehazing
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

low_img = load_image(low, transform).to(device)
high_img = load_image(high, transform).to(device)

#model.train()
#optimizer.zero_grad()
output = model(low_img)
loss = criterion(output, high_img)

#loss.backward()
#optimizer.step()
print(f'Loss: {loss.item():.4f}')
