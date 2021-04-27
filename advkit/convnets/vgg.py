import torch.nn as nn


class VGG(nn.Module):
    default_conv_layers_configs = {
        "vgg16": [2, 2, 3, 3, 3],
        "vgg19": [2, 2, 4, 4, 4]
    }

    def __init__(self, conv_layers_config, input_shape=(3, 32, 32), n_class=10):
        super(VGG, self).__init__()
        self.conv_layers_config = conv_layers_config
        self.input_shape = input_shape
        self.n_class = n_class

        self.block1 = self._get_basic_block(3, 64, self.conv_layers_config[0])
        self.block2 = self._get_basic_block(64, 128, self.conv_layers_config[1])
        self.block3 = self._get_basic_block(128, 256, self.conv_layers_config[2])
        self.block4 = self._get_basic_block(256, 512, self.conv_layers_config[3])
        self.block5 = self._get_basic_block(512, 512, self.conv_layers_config[4])

        post_conv_shape = self.input_shape[1] // 2 ** 5, self.input_shape[2] // 2 ** 5

        self.final_block = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(post_conv_shape[0] * post_conv_shape[1] * 512, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, self.n_class)
        )

    @staticmethod
    def _get_basic_block(in_chans, out_chans, n_layers):
        layers = []
        layers.append(nn.Conv2d(in_chans, out_chans, 3, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_chans))
        layers.append(nn.ReLU())

        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(out_chans, out_chans, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_chans))
            layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.final_block(x)

        return x

    @staticmethod
    def from_default_config(model_type, input_shape=(3, 32, 32), n_class=10):
        return VGG(
            conv_layers_config=VGG.default_conv_layers_configs[model_type],
            input_shape=input_shape,
            n_class=n_class
        )


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

    WEIGHT_PATH = os.path.join(ROOT, "model_weights/cifar_vgg16.pt")
    TRAIN = not os.path.exists(WEIGHT_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(dataset="cifar10", root=DATA_PATH)

    if not TRAIN:
        model = VGG.from_default_config("vgg16")
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
        model = VGG.from_default_config("vgg16")
        model.to(DEVICE)
        epochs = 100
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        best_epoch, best_val_acc = train(model, epochs, train_loader,
                                         loss_fn, optimizer, scheduler, val_loader, DEVICE)

        set_seed(42)
        train_loader = get_dataloader(
            dataset=DATA_PATH,
            root=os.path.join(ROOT, "datasets"),
            train=True,
            train_batch_size=64
        )
        set_seed(42)
        model = VGG.from_default_config("vgg16")
        model.to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        train(model, best_epoch, train_loader, loss_fn, optimizer, scheduler, test_loader, DEVICE)

        torch.save(model.state_dict(), WEIGHT_PATH)
