import torch
import torch.nn as nn

latent_size = 40
g_feature_size = 64
d_feature_size = 64
channel_size = 3
class_num = 24

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size + class_num, g_feature_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_feature_size * 8),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(g_feature_size * 8, g_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_feature_size * 4),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(g_feature_size * 4, g_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_feature_size * 2),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(g_feature_size * 2, g_feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_feature_size),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(g_feature_size, channel_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        

    def forward(self, input_z, input_c):
        input_c = input_c.unsqueeze(2).unsqueeze(3)
        latent = torch.cat((input_z, input_c), 1)
        output = self.conv1(latent)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(3*64*64+class_num, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.discriminator_layer = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.classifier_layer = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid(),
        )

    def forward(self, img, input_c):
        img = img.view(img.size(0), -1)
        latent = torch.cat((img, input_c), 1)
        feature = self.fc1(latent)
        feature = self.fc2(feature)
        feature = self.fc3(feature)

        d_output = self.discriminator_layer(feature).view(-1)
        c_output = self.classifier_layer(feature)

        return d_output, c_output


