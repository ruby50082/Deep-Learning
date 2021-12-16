import torch.nn as nn
class EEGNET_ELU(nn.Module):
    def __init__(self):
        super(EEGNET_ELU, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = (1, 51),
                stride = (1, 1),
                padding = (0, 25),
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 16,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = (2, 1),
                stride = (1, 1),
                groups = 16,
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 32,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ELU(alpha = 0.1),
            nn.AvgPool2d(
                kernel_size = (1, 4),
                stride = (1, 4),
                padding = 0,
            ),
            nn.Dropout(p = 0.5)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = (1, 15),
                stride = (1, 1),
                padding = (0, 7),
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 32,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ELU(alpha = 1.0),
            nn.AvgPool2d(
                kernel_size = (1, 8),
                stride = (1, 8),
                padding = 0,
            ),
            nn.Dropout(p = 0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 736,
                out_features = 2,
                bias = True,
            ),
        )
    
    def forward(self, x):           # [108, 1, 2, 750]   
        x = self.firstconv(x)       # [108, 16, 2, 750]
        x = self.depthwiseConv(x)   # [108, 32, 1, 187]
        x = self.separableConv(x)   # [108, 32, 1, 23]
        x = x.view(x.size(0), -1)   # [108, 736]
        output = self.classifier(x) # [108, 2]

        return output

class EEGNET_RELU(nn.Module):
    def __init__(self):
        super(EEGNET_RELU, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = (1, 51),
                stride = (1, 1),
                padding = (0, 25),
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 16,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = (2, 1),
                stride = (1, 1),
                groups = 16,
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 32,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size = (1, 4),
                stride = (1, 4),
                padding = 0,
            ),
            nn.Dropout(p = 0.5)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = (1, 15),
                stride = (1, 1),
                padding = (0, 7),
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 32,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size = (1, 8),
                stride = (1, 8),
                padding = 0,
            ),
            nn.Dropout(p = 0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 736,
                out_features = 2,
                bias = True,
            ),
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1) #
        output = self.classifier(x)

        return output

class EEGNET_LEAKY_RELU(nn.Module):
    def __init__(self):
        super(EEGNET_LEAKY_RELU, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = (1, 51),
                stride = (1, 1),
                padding = (0, 25),
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 16,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = (2, 1),
                stride = (1, 1),
                groups = 16,
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 32,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(
                kernel_size = (1, 4),
                stride = (1, 4),
                padding = 0,
            ),
            nn.Dropout(p = 0.5)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = (1, 15),
                stride = (1, 1),
                padding = (0, 7),
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 32,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(
                kernel_size = (1, 8),
                stride = (1, 8),
                padding = 0,
            ),
            nn.Dropout(p = 0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 736,
                out_features = 2,
                bias = True,
            ),
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1) #
        output = self.classifier(x)

        return output

class DeepConvNet_ELU(nn.Module):
    def __init__(self):
        super(DeepConvNet_ELU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 25,
                kernel_size = (1, 5),
                stride = 1,
                bias = False,
            ),
            nn.Conv2d(
                in_channels = 25,
                out_channels = 25,
                kernel_size = (2, 1),
                stride = 1,
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 25,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ELU(alpha = 1.0),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2),
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 25,
                out_channels = 50,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 50,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ELU(alpha = 1.0),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 50,
                out_channels = 100,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 100,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ELU(alpha = 1.0),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 100,
                out_channels = 200,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 200,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ELU(alpha = 1.0),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 9200,
                out_features = 2,
                bias = True,
            ),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output

class DeepConvNet_RELU(nn.Module):
    def __init__(self):
        super(DeepConvNet_RELU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 25,
                kernel_size = (1, 5),
                stride = 1,
                bias = False,
            ),
            nn.Conv2d(
                in_channels = 25,
                out_channels = 25,
                kernel_size = (2, 1),
                stride = 1,
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 25,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2),
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 25,
                out_channels = 50,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 50,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 50,
                out_channels = 100,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 100,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 100,
                out_channels = 200,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 200,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 9200,
                out_features = 2,
                bias = True,
            ),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output

class DeepConvNet_LEAKY_RELU(nn.Module):
    def __init__(self):
        super(DeepConvNet_LEAKY_RELU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 25,
                kernel_size = (1, 5),
                stride = 1,
                bias = False,
            ),
            nn.Conv2d(
                in_channels = 25,
                out_channels = 25,
                kernel_size = (2, 1),
                stride = 1,
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 25,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2),
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 25,
                out_channels = 50,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 50,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 50,
                out_channels = 100,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 100,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 100,
                out_channels = 200,
                kernel_size = (1, 5),
                stride = 1,
                padding = (0, 2), ###
                bias = False,
            ),
            nn.BatchNorm2d(
                num_features = 200,
                eps = 1e-05,
                momentum = 0.1,
                affine = True,
                track_running_stats = True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (1, 2),
                stride = (1, 2), ###
                padding = 0,
            ),
            nn.Dropout(p = 0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 9200,
                out_features = 2,
                bias = True,
            ),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output

