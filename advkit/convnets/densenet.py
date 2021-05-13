import torch
import torch.nn as nn

default_configs = {
    "densenet-40-12": {
        "growth_rate": 12,
        "block_config": [12, 12, 12],
        "n_class": 10,
        "imagenet_init": False
    },
    "densenet-100-12": {
        "growth_rate": 12,
        "block_config": [32, 32, 32],
        "n_class": 10,
        "imagenet_init": False
    },
    "densenet-100-24": {
        "growth_rate": 24,
        "block_config": [32, 32, 32],
        "n_class": 10,
        "imagenet_init": False
    },
    "densenet-bc-100-12": {
        "growth_rate": 12,
        "block_config": [16, 16, 16],
        "n_class": 10,
        "imagenet_init": False,
        "bottleneck": True,
        "transition_ratio": 2
    },
    "densenet-bc-250-24": {
        "growth_rate": 24,
        "block_config": [41, 41, 41],
        "n_class": 10,
        "imagenet_init": False,
        "bottleneck": True,
        "transition_ratio": 2
    },
    "densenet-bc-190-40": {
        "growth_rate": 40,
        "block_config": [31, 31, 31],
        "n_class": 10,
        "imagenet_init": False,
        "bottleneck": True,
        "transition_ratio": 2
    }
}


class DenseBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            growth_rate,
            n_blocks,
            bottleneck=False
    ):
        """
        Densely-connected Block
        :param in_channels: number of input feature maps
        :param growth_rate: growth rate
        :param n_blocks: number of blocks
        :param bottleneck: whether to use bottleneck block
        """
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.bottleneck = bottleneck
        self.layer_type = "bottleneck" if self.bottleneck else "standard"

        for i in range(1, n_blocks + 1):
            if i == 1:
                self.add_module(
                    f"{self.layer_type}_layer_{i}",
                    self._get_layer(self.in_channels)
                )
            else:
                self.add_module(
                    f"{self.layer_type}_layer_{i}",
                    self._get_layer(self.in_channels + (i - 1) * self.growth_rate)
                )

    def _get_layer(self, in_channels):

        if self.bottleneck:
            bottleneck_width = 4 * self.growth_rate
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, bottleneck_width, 1, 1, 0, bias=False),
                nn.BatchNorm2d(bottleneck_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_width, self.growth_rate, 3, 1, 1, bias=False)
            )
        else:
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, self.growth_rate, 3, 1, 1, bias=False)
            )

    def forward(self, x):
        inps = x
        for chd in self.children():
            out = chd(inps)
            inps = torch.cat([inps, out], dim=1)
        return inps


class DenseNet(nn.Module):
    default_configs = default_configs

    def __init__(
            self,
            growth_rate,
            block_config,
            n_class,
            in_channels=3,
            imagenet_init=True,
            bottleneck=False,
            transition_ratio=1
    ):

        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.block_config = block_config

        self.n_class = n_class
        self.in_channels = in_channels
        self.imagenet_init = imagenet_init

        self.bottleneck = bottleneck
        self.transition_ratio = transition_ratio
        self.bc_structure = "" + "B" if self.bottleneck else "" + "C" if self.transition_ratio > 1 else ""
        self.init_out_channels = 16 if self.bc_structure != "BC" else 2 * self.growth_rate

        if self.imagenet_init:
            self.feature = nn.Sequential(
                nn.Conv2d(self.in_channels, self.init_out_channels, 7, 2, 3, bias=False),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
        else:
            self.feature = nn.Conv2d(self.in_channels, self.init_out_channels, 3, 1, 1, bias=False)

        out_channels = self.init_out_channels
        for i, n_blocks in enumerate(self.block_config):
            self.add_module(
                f"dense_layer_{i + 1}",
                DenseBlock(out_channels, self.growth_rate, n_blocks, self.bottleneck)
            )
            out_channels += n_blocks * self.growth_rate
            if i < len(self.block_config) - 1:
                self.add_module(f"transition_layer_{i + 1}", self._get_transition_block(out_channels))
                out_channels = out_channels // self.transition_ratio

        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels, self.n_class)
        )

    def _get_transition_block(self, in_channels):

        transition_width = in_channels // self.transition_ratio
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, transition_width, 1, 1, 0, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        for chd in self.children():
            x = chd(x)
        return x

    @staticmethod
    def from_default_config(model_type):
        return DenseNet(**DenseNet.default_configs[model_type])


if __name__ == "__main__":
    import os
    from advkit.utils.models import *
    from advkit.utils.data import get_dataloader
    from torch.optim import SGD, lr_scheduler

    ROOT = os.path.expanduser("~/advkit")
    DATA_PATH = os.path.join(ROOT, "datasets")

    CHECKPOINT_PATH = os.path.join(ROOT, "model_weights/cifar10_densenet-bc-100-12.pt")
    TRAIN = not os.path.exists(CHECKPOINT_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentation = False
    testloader = get_dataloader(dataset="cifar10", root=DATA_PATH, augmentation=augmentation)

    if not TRAIN:
        model = DenseNet.from_default_config("densenet-bc-100-12")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
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
        model = DenseNet.from_default_config("densenet-bc-100-12")
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
