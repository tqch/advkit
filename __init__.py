if __name__ == "__main__":

    import os
    import torch
    import torch.nn as nn
    from convnets.vgg import VGG16
    from utils.train import train,set_seed
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from torchvision import datasets,transforms
    from utils.data import CIFAR10Dataset
    from torch.optim import Adam

    TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    ROOT = "./datasets"
    DATAFILE = "cifar-10-python.tar.gz"
    DOWNLOAD = not os.path.exists(os.path.join(ROOT,DATAFILE))

    WEIGHT_PATH = "./model_weights/cifar_vgg16.pt"
    TRAIN = not os.path.exists(WEIGHT_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset = datasets.CIFAR10(root=ROOT, download=DOWNLOAD, train=False, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=1024, shuffle=False)

    if not TRAIN:
        model = VGG16()
        model.load_state_dict(torch.load(WEIGHT_PATH,map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for x,y in testloader:
                out = model(x.to(DEVICE))
                pred = out.max(dim=1)[1]
                test_correct += (pred==y.to(DEVICE)).sum().item()
                test_total += x.size(0)
        print("Test accuracy is %.3f" % (test_correct/test_total))

    else:
        trainset = datasets.CIFAR10(root=ROOT, download=DOWNLOAD, train=True)
        trainset_, valset_ = train_test_split(trainset, test_size=0.1, random_state=42)
        trainset_, valset_ = CIFAR10Dataset(trainset_, transform=TRANSFORM), CIFAR10Dataset(valset_)

        set_seed(42)
        trainloader = DataLoader(trainset_, shuffle=True, batch_size=256)
        valloader = DataLoader(valset_, shuffle=False, batch_size=1024)

        set_seed(42)
        model = VGG16()
        model.to(DEVICE)

        epochs = 50
        loss_fn = nn.CrossEntropyLoss()
        opt = Adam(model.parameters())

        best_epoch, best_val_acc = train(model, epochs, trainloader, loss_fn, opt, valloader, DEVICE)

        set_seed(42)
        trainloader = DataLoader(CIFAR10Dataset(trainset, transform=TRANSFORM), shuffle=True, batch_size=256)
        valloader = DataLoader(testset, shuffle=False, batch_size=1024)

        set_seed(42)
        model = VGG16()
        model.to(DEVICE)

        loss_fn = nn.CrossEntropyLoss()
        opt = Adam(model.parameters())

        train(model, best_epoch, trainloader, loss_fn, opt, valloader, DEVICE)

        torch.save(model.state_dict(), "./model_weights/cifar_vgg16.pt")