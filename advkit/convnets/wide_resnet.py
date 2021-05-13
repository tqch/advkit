import torch.nn as nn
import torch.nn.functional as F

default_configs = {
    "WRN-28-10": {
        "block_type": "basic",
        "layer_config": [4, 4, 4],
        "widening_factor": 10,
        "start_channels": 16,
        "n_class": 10,
        "dropout_prob": 0.3
    },
    "WRN-28-12": {
        "block_type": "basic",
        "layer_config": [4, 4, 4],
        "widening_factor": 12,
        "start_channels": 16,
        "n_class": 100,
        "dropout_prob": 0.3
    },
    "WRN-40-10": {
        "block_type": "basic",
        "layer_config": [6, 6, 6],
        "widening_factor": 10,
        "start_channels": 16,
        "n_class": 100,
        "dropout_prob": 0.3
    },
    "WRN-50-2-bottleneck": {
        "block_type": "bottleneck",
        "layer_config": [3, 4, 6, 3],
        "widening_factor": 2,
        "start_channels": 64,
        "n_class": 1000
    }
}


class BasicBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            widening_factor=1,
            dropout_prob=0.5,
            is_initial=False
    ):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels * (widening_factor if not is_initial else 1)
        self.out_channels = out_channels * widening_factor
        self.widen_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.downsample = in_channels != out_channels
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, self.downsample + 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1, bias=False)
        self.projection = nn.Conv2d(
            self.in_channels, self.out_channels, 1, self.downsample + 1, 0, bias=False
        ) if self.downsample or is_initial else nn.Identity()

    def forward(self, x):
        skip_connection = self.projection(x)
        conv1 = self.conv1(self.relu(self.bn1(x)))
        if self.dropout_prob > 0:
            conv1 = F.dropout(conv1, p=self.dropout_prob, training=self.training, inplace=True)
        conv2 = self.conv2(self.relu(self.bn2(conv1)))
        return conv2 + skip_connection


class BottleneckBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            widening_factor=1,
            dropout_prob=0,
            is_initial=False
    ):
        super(BottleneckBlock, self).__init__()
        self.in_channels = in_channels * (widening_factor if not is_initial else 1)
        self.out_channels = out_channels * (widening_factor if not is_initial else 1)
        self.neck_channels = self.out_channels // 4
        self.widen_factor = widening_factor
        self.downsample = in_channels != out_channels and not is_initial
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.neck_channels, 1, self.downsample + 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.neck_channels)
        self.conv2 = nn.Conv2d(self.neck_channels, self.neck_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = nn.Conv2d(self.neck_channels, self.out_channels, 1, 1, 0, bias=False)
        self.projection = nn.Conv2d(
            self.in_channels, self.out_channels, 1, self.downsample + 1, 0, bias=False
        ) if self.downsample or is_initial else nn.Identity()

    def forward(self, x):
        skip_connection = self.projection(x)
        conv1 = self.conv1(self.relu(self.bn1(x)))
        conv2 = self.conv2(self.relu(self.bn2(conv1)))
        conv3 = self.conv3(self.relu(self.bn3(conv2)))
        return conv3 + skip_connection


class WideResNet(nn.Module):
    block_dict = {
        "basic": BasicBlock,
        "bottleneck": BottleneckBlock
    }
    default_configs = default_configs

    def __init__(
            self,
            start_channels,
            block_type,
            layer_config,
            widening_factor,
            dropout_prob=0,
            n_class=10
    ):
        super(WideResNet, self).__init__()
        self.n_class = n_class
        self.block = self.block_dict[block_type]
        if start_channels == 16:
            self.feature = nn.Conv2d(3, start_channels, 3, 1, 1, bias=False)
        else:
            self.feature = nn.Sequential(
                nn.Conv2d(3, start_channels, 7, 2, 3, bias=False),
                nn.MaxPool2d(2),
                self._make_layer(start_channels, start_channels, 1, layer_config[0])
            )
            layer_config = layer_config[1:]
        self.widening_factor = widening_factor
        for i in range(3):
            self.add_module(
                f"layer{i + 1}",
                self._make_layer(
                    start_channels * 2 ** (i - 1 + (i == 0)),
                    start_channels * 2 ** i,
                    widening_factor,
                    layer_config[0],
                    dropout_prob,
                    is_initial=(i == 0)
                ))
        self.bn = nn.BatchNorm2d(4 * start_channels * widening_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4 * start_channels * widening_factor, n_class)

    def _make_layer(
            self,
            in_channels,
            out_channels,
            widening_factor=1,
            n_blocks=1,
            dropout_prob=0.5,
            is_initial=False
    ):
        blocks = [self.block(
            in_channels,
            out_channels,
            widening_factor,
            dropout_prob,
            is_initial
        )]
        for i in range(1, n_blocks):
            blocks.append(self.block(
                out_channels,
                out_channels,
                widening_factor,
                dropout_prob
            ))
        return nn.Sequential(*blocks)

    def forward(self, x):
        feature = self.feature(x)
        layer1 = self.layer1(feature)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        prepool = self.relu(self.bn(layer3))
        pool = self.avgpool(prepool)
        out = self.classifier(pool.flatten(start_dim=1))
        return out

    @staticmethod
    def from_default_config(model_type):
        return WideResNet(**WideResNet.default_configs[model_type])


if __name__ == "__main__":
    import os
    from advkit.utils.models import *
    from advkit.utils.data import get_dataloader
    from torch.optim import SGD, lr_scheduler

    ROOT = os.path.expanduser("~/advkit")
    DATA_PATH = os.path.join(ROOT, "datasets")

    WEIGHTS_PATH = os.path.join(ROOT, "model_weights/cifar10_wrn-28-10.pt")
    TRAIN = not os.path.exists(WEIGHTS_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentation = False
    testloader = get_dataloader(dataset="cifar10", root=DATA_PATH, augmentation=augmentation)

    if not TRAIN:
        model = WideResNet.from_default_config("WRN-28-10")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE)["model"])
        model.to(DEVICE)
        evaluate(model, testloader, device=DEVICE)
    else:
        set_seed(42)
        trainloader = get_dataloader(
            dataset="cifar10",
            root=DATA_PATH,
            train=True,
            train_batch_size=128,
            augmentation=augmentation
        )

        model = WideResNet.from_default_config("WRN-28-10")
        model.to(DEVICE)
        epochs = 200
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), **OPTIMIZER_CONFIGS["cifar10"])
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lambda epochs: 0.1 * 0.2 ** ((epochs >= 60) + (epochs >= 120) + (epochs >= 160))
        )

        best_epoch, best_val_acc = train(
            model,
            epochs,
            trainloader,
            loss_fn,
            optimizer,
            scheduler,
            testloader,
            num_eval_batches=-1,
            checkpoint_path=WEIGHTS_PATH,
            device=DEVICE
        )
        if not os.path.exists(os.path.dirname(WEIGHTS_PATH)):
            os.makedirs(os.path.dirname(WEIGHTS_PATH))
        torch.save(model.state_dict(), WEIGHTS_PATH)
