import torch.nn as nn

cifar10_feature_config = [
            {
                "in_channels": 3,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "bias": False
            },
            {"max_pool": False}
        ]

imagenet_feature_config = [
            {
                "in_channels": 3,
                "out_channels": 64,
                "kernel_size": 7,
                "stride": 2,
                "padding": 3,
                "bias": False
            },
            {"max_pool": True}
        ]

default_configs = {
    "resnet18": {
        "feature_config": imagenet_feature_config,
        "block_type": "basic",
        "start_channels": 64,
        "n_class": 1000,
        "n_blocks": [2, 2, 2, 2]
    },
    "resnet20": {
        "feature_config": cifar10_feature_config,
        "block_type": "basic",
        "start_channels": 16,
        "n_class": 10,
        "n_blocks": [3, 3, 3]
    },
    "resnet32": {
        "feature_config": cifar10_feature_config,
        "block_type": "basic",
        "start_channels": 16,
        "n_class": 10,
        "n_blocks": [5, 5, 5]
    },
    "resnet34": {
        "feature_config": imagenet_feature_config,
        "block_type": "basic",
        "start_channels": 64,
        "n_class": 1000,
        "n_blocks": [3, 4, 6, 3]
    },
    "resnet44": {
        "feature_config": cifar10_feature_config,
        "block_type": "basic",
        "start_channels": 16,
        "n_class": 10,
        "n_blocks": [7, 7, 7]
    },
    "resnet50": {
        "feature_config": imagenet_feature_config,
        "block_type": "bottleneck",
        "start_channels": 64,
        "n_class": 1000,
        "n_blocks": [3, 4, 6, 3]
    },
    "resnet56": {
        "feature_config": cifar10_feature_config,
        "block_type": "basic",
        "start_channels": 16,
        "n_class": 10,
        "n_blocks": [9, 9, 9]
    },
    "resnet101": {
        "feature_config": imagenet_feature_config,
        "block_type": "bottleneck",
        "start_channels": 64,
        "n_class": 1000,
        "n_blocks": [3, 4, 23, 3]
    },
    "resnet110": {
        "feature_config": cifar10_feature_config,
        "block_type": "basic",
        "start_channels": 16,
        "n_class": 10,
        "n_blocks": [18, 18, 18]
    },
    "resnet152": {
        "feature_config": imagenet_feature_config,
        "block_type": "bottleneck",
        "start_channels": 64,
        "n_class": 1000,
        "n_blocks": [3, 8, 36, 3]
    }
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, downsample=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        stride = 1
        out_channels = in_channels
        self.skip_connection = nn.Identity()
        
        if self.downsample:
            stride = 2
            out_channels = 2 * in_channels
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        projection = self.skip_connection(x)
        x = self.conv(x)
        x = self.activation(x + projection)
        return x


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, downsample):
        super(BottleneckBlock, self).__init__()

        self.downsample = downsample
        stride = 1
        neck_channels = in_channels // self.expansion
        out_channels = in_channels
        self.skip_connection = nn.Identity()
        if self.downsample:
            stride = 2
            neck_channels = in_channels * 2 // self.expansion
            out_channels = 2 * in_channels
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, neck_channels, 1, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_channels, neck_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        projection = self.skip_connection(x)
        x = self.conv(x)
        x = self.activation(x + projection)
        return x


class ResNet(nn.Module):
    block = {
        "basic": BasicBlock,
        "bottleneck": BottleneckBlock
    }
    default_configs = default_configs

    def __init__(
            self,
            feature_config,
            block_type,
            start_channels,
            n_class,
            n_blocks
    ):
        super(ResNet, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(**feature_config[0]),
            nn.BatchNorm2d(feature_config[0]["out_channels"]),
            nn.ReLU()
        )
        if feature_config[1]["max_pool"]:
            self.feature.add_module("max_pool", nn.MaxPool2d(3, 2, 1))

        self.add_module(
            "residual_layer_1",
            self._make_layer(block_type, start_channels, n_blocks[0], include_downsample=False)
        )

        for i in range(1, len(n_blocks)):
            self.add_module(
                f"residual_layer_{i + 1}",
                self._make_layer(block_type, 2 ** (i - 1) * start_channels, n_blocks[i])
            )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2 ** i * start_channels, n_class)
        )

    def _make_layer(
            self,
            block_type,
            in_channels,
            n_blocks,
            include_downsample=True
    ):

        block = self.block[block_type]
        blocks = [block(in_channels, downsample=include_downsample)]
        out_channels = in_channels
        if include_downsample:
            out_channels = 2 * in_channels
        for _ in range(1, n_blocks):
            blocks.append(block(out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        for chd in self.children():
            x = chd(x)
        return x

    @staticmethod
    def from_default_config(model_type):
        return ResNet(**ResNet.default_configs[model_type])


if __name__ == "__main__":
    import os
    from advkit.utils.models import *
    from advkit.utils.data import get_dataloader
    from torch.optim import SGD, lr_scheduler

    ROOT = os.path.expanduser("~/advkit")
    DATA_PATH = os.path.join(ROOT, "datasets")

    CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints/cifar10_resnet56.pt")
    TRAIN = not os.path.exists(CHECKPOINT_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentation = True
    testloader = get_dataloader(dataset="cifar10", root=DATA_PATH, augmentation=augmentation)

    if not TRAIN:
        model = ResNet.from_default_config("resnet56")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_weights"])
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

        model = ResNet.from_default_config("resnet56")
        model.to(DEVICE)
        epochs = 200
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), **OPTIMIZER_CONFIGS["cifar10"])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

        best_epoch, best_val_acc = train(
            model,
            epochs,
            trainloader,
            loss_fn,
            optimizer,
            scheduler,
            testloader,
            num_eval_batches=-1,
            checkpoint_path=CHECKPOINT_PATH,
            device=DEVICE
        )
