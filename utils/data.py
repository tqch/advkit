from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR10Dataset(Dataset):

    def __init__(self,data,transform=transforms.ToTensor()):
        self.data = data
        self.transform = transform
        self.length = len(data)

    def __getitem__(self,idx):
        return self.transform(self.data[idx][0]),self.data[idx][1]

    def __len__(self):
        return self.length

