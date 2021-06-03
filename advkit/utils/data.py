from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from sklearn.model_selection import train_test_split

DATASET_CONFIGS = {
    "mnist": {
        "data_folder": "MNIST",
        "importer": datasets.MNIST,
        "transform_train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            transforms.RandomCrop(size=28, padding=4)
        ]),
        "transform_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
        "mean": (0.1307, ),
        "std": (0.3081, ),
        "batch_size": 128
    },
    "cifar10": {
        "data_folder": "cifar-10-python.tar.gz",
        "importer": datasets.CIFAR10,
        "transform_train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=32, padding=4)
        ]),
        "transform_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ]),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "batch_size": 128
    }
}

ROOT = os.path.expanduser("~/advkit")
DATA_PATH = os.path.join(ROOT, "datasets")
CHECKPOINT_FOLDER = os.path.join(ROOT, "checkpoints")


class PyTorchDataset(Dataset):

    def __init__(self, data, transform=transforms.ToTensor()):
        self.data = data
        self.transform = transform
        self.length = len(data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx][0]), self.data[idx][1]

    def __len__(self):
        return self.length


def get_dataloader(
        dataset,
        root=DATA_PATH,
        download=None,
        train=False,
        transform_train=None,
        transform_test=None,
        val_size=None,
        random_state=42,
        train_batch_size=None,
        val_batch_size=128,
        test_batch_size=128,
        augmentation=False
):
    configs = DATASET_CONFIGS[dataset]
    data_folder = configs["data_folder"]
    importer = configs["importer"]
    train_batch_size = train_batch_size or configs["batch_size"]
    if not os.path.exists(root):
        os.makedirs(root)
    if download is None:
        download = not os.path.exists(os.path.join(root, data_folder))
    if not train:
        if transform_test is None:
            transform_test = configs["transform_test"]\
                if augmentation else transforms.ToTensor()
        dataset = importer(
            root=root,
            train=False,
            transform=transform_test,
            download=download
        )
        return DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    else:
        if transform_train is None:
            transform_train = configs["transform_train"]\
                if augmentation else transforms.ToTensor()
        if transform_test is None:
            transform_test = configs["transform_test"]\
                if augmentation else transforms.ToTensor()
        if val_size is None:
            dataset = importer(
                root=root,
                train=True,
                transform=transform_train,
                download=download
            )
            return DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
        else:
            dataset = importer(root=root, train=True, download=download)
            trainset, valset = train_test_split(
                dataset,
                test_size=val_size,
                random_state=random_state
            )
            trainset = PyTorchDataset(trainset, transform=transform_train)
            valset = PyTorchDataset(valset, transform=transform_test)
            return (
                DataLoader(trainset, batch_size=train_batch_size, shuffle=True),
                DataLoader(valset, batch_size=val_batch_size, shuffle=False)
            )
