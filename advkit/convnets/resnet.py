import torch.nn as nn

cifar10_initial_block_config = [
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

imagenet_initial_block_config = [
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
        "initial_block_config": imagenet_initial_block_config,
        "block_type": "basic",
        "start_chans": 64,
        "n_class": 1000,
        "n_blocks": [2, 2, 2, 2]
    },
    "resnet20": {
        "initial_block_config": cifar10_initial_block_config,
        "block_type": "basic",
        "start_chans": 16,
        "n_class": 10,
        "n_blocks": [3, 3, 3]
    },
    "resnet32": {
        "initial_block_config": cifar10_initial_block_config,
        "block_type": "basic",
        "start_chans": 16,
        "n_class": 10,
        "n_blocks": [5, 5, 5]
    },
    "resnet34": {
        "initial_block_config": imagenet_initial_block_config,
        "block_type": "basic",
        "start_chans": 64,
        "n_class": 1000,
        "n_blocks": [3, 4, 6, 3]
    },
    "resnet44": {
        "initial_block_config": cifar10_initial_block_config,
        "block_type": "basic",
        "start_chans": 16,
        "n_class": 10,
        "n_blocks": [7, 7, 7]
    },
    "resnet50": {
        "initial_block_config": imagenet_initial_block_config,
        "block_type": "bottleneck",
        "start_chans": 64,
        "n_class": 1000,
        "n_blocks": [3, 4, 6, 3]
    },
    "resnet56": {
        "initial_block_config": cifar10_initial_block_config,
        "block_type": "basic",
        "start_chans": 16,
        "n_class": 10,
        "n_blocks": [9, 9, 9]
    },
    "resnet101": {
        "initial_block_config": imagenet_initial_block_config,
        "block_type": "bottleneck",
        "start_chans": 64,
        "n_class": 1000,
        "n_blocks": [3, 4, 23, 3]
    },
    "resnet110": {
        "initial_block_config": cifar10_initial_block_config,
        "block_type": "basic",
        "start_chans": 16,
        "n_class": 10,
        "n_blocks": [18, 18, 18]
    },
    "resnet152": {
        "initial_block_config": imagenet_initial_block_config,
        "block_type": "bottleneck",
        "start_chans": 64,
        "n_class": 1000,
        "n_blocks": [3, 8, 36, 3]
    }
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chans, downsample=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        stride = 1
        out_chans = in_chans
        self.skip_connection = nn.Identity()
        if self.downsample:
            stride = 2
            out_chans = 2 * in_chans
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 1, stride, bias=False),
                nn.BatchNorm2d(out_chans)
            )

        self.plain_block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chans)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        projection = self.skip_connection(x)
        x = self.plain_block(x)
        x = self.activation(x + projection)
        return x


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_chans, downsample):
        super(BottleneckBlock, self).__init__()

        self.downsample = downsample
        stride = 1
        neck_chans = in_chans // self.expansion
        out_chans = in_chans
        self.skip_connection = nn.Identity()
        if self.downsample:
            stride = 2
            neck_chans = in_chans * 2 // self.expansion
            out_chans = 2 * in_chans
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 1, stride, bias=False),
                nn.BatchNorm2d(out_chans)
            )

        self.plain_block = nn.Sequential(
            nn.Conv2d(in_chans, neck_chans, 1, stride, 1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(neck_chans, neck_chans, 3, 1, 1, bias=False),
            nn.BatchNorm2d(neck_chans),
            nn.ReLU(),
            nn.Conv2d(neck_chans, out_chans, 1, 1, bias=False),
            nn.BatchNorm2d(out_chans)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        projection = self.skip_connection(x)
        x = self.plain_block(x)
        x = self.activation(x + projection)
        return x


class ResNet(nn.Module):
    block = {
        "basic": BasicBlock,
        "bottleneck": BottleneckBlock
    }
    default_configs = default_configs

    def __init__(self, initial_block_config, block_type, start_chans, n_class, n_blocks):
        super(ResNet, self).__init__()

        self.initial_block = nn.Sequential(
            nn.Conv2d(**initial_block_config[0]),
            nn.BatchNorm2d(initial_block_config[0]["out_channels"]),
            nn.ReLU()
        )
        if initial_block_config[1]["max_pool"]:
            self.initial_block.add_module("max_pool", nn.MaxPool2d(3, 2, 1))

        self.add_module(
            "residual_block_1",
            self._compose_blocks(block_type, start_chans, n_blocks[0], include_downsample=False)
        )

        for i in range(1, len(n_blocks)):
            self.add_module(
                f"residual_block_{i + 1}",
                self._compose_blocks(block_type, 2 ** (i - 1) * start_chans, n_blocks[i])
            )

        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2 ** i * start_chans, n_class)
        )

    def _compose_blocks(self, block_type, in_chans, n_blocks, include_downsample=True):

        block = self.block[block_type]
        blocks = [block(in_chans, downsample=include_downsample)]
        out_chans = in_chans
        if include_downsample:
            out_chans = 2 * in_chans
        for _ in range(1, n_blocks):
            blocks.append(block(out_chans))
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
    from torchvision import transforms
    from torch.optim import SGD,lr_scheduler

    ROOT = os.path.expanduser("~/advkit")
    DATA_PATH = os.path.join(ROOT, "datasets")
    TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.125, 0.125), scale=(0.875, 1.125)),
        transforms.ToTensor()
    ])

    WEIGHT_PATH = os.path.join(ROOT, "model_weights/cifar_resnet56.pt")
    TRAIN = not os.path.exists(WEIGHT_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(dataset="cifar10", root=DATA_PATH)

    if not TRAIN:
        model = ResNet.from_default_config("resnet56")
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
        model.to(DEVICE)
        evaluate(model, test_loader, device=DEVICE)
    else:
        set_seed(42)
        train_loader, val_loader = get_dataloader(
            "cifar10",
            root=DATA_PATH,
            train=True,
            transform=TRANSFORM,
            val_size=0.1,
            train_batch_size=64
        )
        set_seed(42)
        model = ResNet.from_default_config("resnet56")
        model.to(DEVICE)
        epochs = 100
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        best_epoch, best_val_acc = train(model, epochs, train_loader,
                                         loss_fn, optimizer, scheduler, val_loader, DEVICE)

        set_seed(42)
        train_loader = get_dataloader(
            dataset="cifar10",
            root=DATA_PATH,
            train=True,
            transform=TRANSFORM,
            train_batch_size=64
        )
        set_seed(42)
        model = ResNet.from_default_config("resnet56")
        model.to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        train(model, best_epoch, train_loader, loss_fn, optimizer, scheduler, test_loader, DEVICE)

        torch.save(model.state_dict(), WEIGHT_PATH)