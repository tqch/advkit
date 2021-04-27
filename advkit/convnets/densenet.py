import torch
import torch.nn as nn

default_configs = {
    "densenet-40-12": {
        "growth_rate": 12,
        "all_layers": [12, 12, 12],
        "n_class": 10,
        "imagenet_init": False
    },
    "densenet-100-12": {
        "growth_rate": 12,
        "all_layers": [32, 32, 32],
        "n_class": 10,
        "imagenet_init": False
    },
    "densenet-100-24": {
        "growth_rate": 24,
        "all_layers": [32, 32, 32],
        "n_class": 10,
        "imagenet_init": False
    },
    "densenet-bc-100-12": {
        "growth_rate": 12,
        "all_layers": [16, 16, 16],
        "n_class": 10,
        "imagenet_init": False,
        "bottleneck": True,
        "transition_ratio": 2
    },
    "densenet-bc-250-24": {
        "growth_rate": 24,
        "all_layers": [41, 41, 41],
        "n_class": 10,
        "imagenet_init": False,
        "bottleneck": True,
        "transition_ratio": 2
    },
    "densenet-bc-190-40": {
        "growth_rate": 40,
        "all_layers": [31, 31, 31],
        "n_class": 10,
        "imagenet_init": False,
        "bottleneck": True,
        "transition_ratio": 2
    }
}


class DenseBlock(nn.Module):

    def __init__(self, in_chans, growth_rate, n_layers, bottleneck=False):
        """
        Densely-connected Block
        :param in_chans: number of input feature maps
        :param growth_rate: growth rate
        :param n_layers: number of layers
        :param bottleneck: whether to use bottleneck block
        """
        super(DenseBlock, self).__init__()
        self.in_chans = in_chans
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.bottleneck = bottleneck
        self.layer_type = "bottleneck" if self.bottleneck else "standard"

        for i in range(1, n_layers + 1):
            if i == 1:
                self.add_module(f"{self.layer_type}_layer_{i}", self._get_layer(self.in_chans))
            else:
                self.add_module(f"{self.layer_type}_layer_{i}",
                                self._get_layer(self.in_chans + (i - 1) * self.growth_rate))

    def _get_layer(self, in_chans):

        if self.bottleneck:
            bottleneck_width = 4 * self.growth_rate
            return nn.Sequential(
                nn.BatchNorm2d(in_chans),
                nn.ReLU(),
                nn.Conv2d(in_chans, bottleneck_width, 1, 1, 0, bias=False),
                nn.BatchNorm2d(bottleneck_width),
                nn.ReLU(),
                nn.Conv2d(bottleneck_width, self.growth_rate, 3, 1, 1, bias=False)
            )
        else:
            return nn.Sequential(
                nn.BatchNorm2d(in_chans),
                nn.ReLU(),
                nn.Conv2d(in_chans, self.growth_rate, 3, 1, 1, bias=False)
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
            all_layers,
            n_class,
            in_chans=3,
            imagenet_init=True,
            bottleneck=False,
            transition_ratio=1
    ):

        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.all_layers = all_layers

        self.n_class = n_class
        self.in_chans = in_chans
        self.imagenet_init = imagenet_init

        self.bottleneck = bottleneck
        self.transition_ratio = transition_ratio
        self.bc_structure = "" + "B" if self.bottleneck else "" + "C" if self.transition_ratio > 1 else ""
        self.init_out_chans = 16 if self.bc_structure != "BC" else 2 * self.growth_rate

        if self.imagenet_init:
            self.initial_block = nn.Sequential(
                nn.Conv2d(self.in_chans, self.init_out_chans, 7, 2, 3, bias=False),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
        else:
            self.initial_block = nn.Conv2d(self.in_chans, self.init_out_chans, 3, 1, 1, bias=False)

        out_chans = self.init_out_chans
        for i, n_layers in enumerate(self.all_layers):
            self.add_module(
                f"dense_block_{i + 1}",
                DenseBlock(out_chans, self.growth_rate, n_layers, self.bottleneck)
            )
            out_chans += n_layers * self.growth_rate
            if i < len(self.all_layers) - 1:
                self.add_module(f"transition_block_{i + 1}", self._get_transition_block(out_chans))
                out_chans = out_chans // self.transition_ratio

        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(out_chans, self.n_class)
        )

    def _get_transition_block(self, in_chans):

        transition_width = in_chans // self.transition_ratio
        return nn.Sequential(
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, transition_width, 1, 1, 0, bias=False),
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
    from torchvision import transforms
    from torch.optim import SGD, lr_scheduler

    ROOT = os.path.expanduser("~/advkit")
    DATA_PATH = os.path.join(ROOT, "datasets")
    TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.125, 0.125), scale=(0.875, 1.125)),
        transforms.ToTensor()
    ])

    WEIGHT_PATH = os.path.join(ROOT, "model_weights/cifar_densenet-bc-100-12.pt")
    TRAIN = not os.path.exists(WEIGHT_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(dataset="cifar10", root=DATA_PATH)

    if not TRAIN:
        model = DenseNet.from_default_config("densenet-bc-100-12")
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
        model.to(DEVICE)
        evaluate(model, test_loader, device=DEVICE)
    else:
        set_seed(42)
        train_loader, val_loader = get_dataloader(
            "cifar10",
            root=DATA_PATH,
            train=True,
            val_size=0.1,
            train_batch_size=64
        )
        set_seed(42)
        model = DenseNet.from_default_config("densenet-bc-100-12")
        model.to(DEVICE)
        epochs = 100
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        best_epoch, best_val_acc = train(model, epochs, train_loader, loss_fn, optimizer, scheduler, val_loader, DEVICE)

        set_seed(42)
        train_loader = get_dataloader(
            dataset="cifar10",
            root=DATA_PATH,
            train=True,
            train_batch_size=64
        )
        set_seed(42)
        model = DenseNet.from_default_config("densenet-bc-100-12")
        model.to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        train(model, best_epoch, train_loader, loss_fn, optimizer, scheduler, test_loader, DEVICE)

        torch.save(model.state_dict(), WEIGHT_PATH)
