import timm
import torch.nn as nn
import torch.nn.functional as F

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PreTrained(nn.Module):
    def __init__(self):
        super(PreTrained, self).__init__()
        self.pt_model = timm.create_model(
            'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
            pretrained=True,
            num_classes=0
        )

        for param in self.pt_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.pt_model.num_features, nclasses)

    def forward(self, x):
        features = self.pt_model(x)
        return self.classifier(features)
