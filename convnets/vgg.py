import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):

    @staticmethod
    def _get_basic_block(in_chans,out_chans,n_layers):

        layers = []

        layers.append(nn.Conv2d(in_chans, out_chans, 3, 1, 1))
        layers.append(nn.BatchNorm2d(out_chans))
        layers.append(nn.ReLU())

        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(out_chans, out_chans, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_chans))
            layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(2))

        return nn.Sequential(*layers)

    def final_layers(self):
        return nn.Sequential(
            self.fc1,
            nn.Dropout(0.5),
            nn.ReLU(),
            self.fc2,
            nn.Dropout(0.5),
            nn.ReLU(),
            self.fc3
        )


class VGG16(VGG):

    def __init__(self, input_shape=(1, 32, 32)):
        super(VGG16, self).__init__()
        self.input_shape = input_shape

        self.block1 = self._get_basic_block(3, 64, 2)
        self.block2 = self._get_basic_block(64, 128, 2)
        self.block3 = self._get_basic_block(128, 256, 3)
        self.block4 = self._get_basic_block(256, 512, 3)
        self.block5 = self._get_basic_block(512, 512, 3)

        post_conv_size = (input_shape[1] // 2 ** 5, input_shape[2] // 2 ** 5)

        self.fc1 = nn.Linear(post_conv_size[0] * post_conv_size[1] * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x).flatten(start_dim=1)
        x = F.dropout(F.relu(self.fc1(x)),0.5)
        x = F.dropout(F.relu(self.fc2(x)),0.5)
        x = self.fc3(x)

        return x


class VGG19(VGG):

    def __init__(self,input_shape=(1,32,32)):
        super(VGG19,self).__init__()
        self.input_shape = input_shape

        self.block1 = self._get_basic_block(3,64,2)
        self.block2 = self._get_basic_block(64,128,2)
        self.block3 = self._get_basic_block(128,256,4)
        self.block4 = self._get_basic_block(256,512,4)
        self.block5 = self._get_basic_block(512,512,4)

        post_conv_size = (input_shape[1]//2**5,input_shape[2]//2**5)

        self.fc1 = nn.Linear(post_conv_size[0]*post_conv_size[1]*512,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)

    def forward(self,x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x).flatten(start_dim=1)
        x = F.dropout(F.relu(self.fc1(x)),0.5)
        x = F.dropout(F.relu(self.fc2(x)),0.5)
        x = self.fc3(x)

        return x
