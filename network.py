import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStreamIQA(nn.Module):
    def __init__(self):
        super(TwoStreamIQA, self).__init__()
        # RGB layer
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.convP1 = nn.Conv2d(in_channels=256, out_channels=9 * 24, kernel_size=1, stride=1, padding=0)

        # gradient layer
        self.conv0_gra = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv1_gra = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2_gra = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_gra = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.convP2 = nn.Conv2d(in_channels=256, out_channels=9 * 24, kernel_size=1, stride=1, padding=0)

        # FC layer
        self.fc1 = nn.Linear(432, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, input):
        x_RGB = input[0].view(-1, input[0].size(-3), input[0].size(-2), input[0].size(-1))
        x_gra = input[1].view(-1, input[1].size(-3), input[1].size(-2), input[1].size(-1))

        # RGB region
        conv0 = F.relu(self.conv0(x_RGB))
        pool0 = F.max_pool2d(conv0, (2, 2), stride=2)

        conv1 = F.relu(self.conv1(pool0))
        pool1 = F.max_pool2d(conv1, (2, 2), stride=2)

        conv2 = F.relu(self.conv2(pool1))
        pool2 = F.max_pool2d(conv2, (2, 2), stride=2)

        conv3 = F.relu(self.conv3(pool2))
        pool3 = F.max_pool2d(conv3, (2, 2), stride=2)

        conv4 = F.relu(self.convP1(pool3))
        pool4 = F.max_pool2d(conv4, (2, 2), stride=2)

        # gradient region
        conv0_gra = F.relu(self.conv0(x_gra))
        pool0_gra = F.max_pool2d(conv0_gra, (2, 2), stride=2)

        conv1_gra = F.relu(self.conv1(pool0_gra))
        pool1_gra = F.max_pool2d(conv1_gra, (2, 2), stride=2)

        conv2_gra = F.relu(self.conv2(pool1_gra))
        pool2_gra = F.max_pool2d(conv2_gra, (2, 2), stride=2)

        conv3_gra = F.relu(self.conv3(pool2_gra))
        pool3_gra = F.max_pool2d(conv3_gra, (2, 2), stride=2)

        conv4_gra = F.relu(self.convP1(pool3_gra))
        pool4_gra = F.max_pool2d(conv4_gra, (2, 2), stride=2)

        pool4 = pool4.squeeze(3).squeeze(2)
        pool4_gra = pool4_gra.squeeze(3).squeeze(2)

        two_stream = torch.cat((pool4, pool4_gra), 1)

        q = self.fc1(two_stream)
        q = self.fc2(q)
        q = self.fc3(q)

        return q


