import os
import torch
import torch.nn as nn
from advkit.utils.models import set_seed
from advkit.utils.data import get_dataloader, DATA_PATH, WEIGHTS_FOLDER
from advkit.attacks.pgd import PGD
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adv_train(
        model,
        optimizer,
        scheduler=None,
        epochs=200,
        eps=4 / 255,
        step_size=2 / 255,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        targeted=False,
        loss_fn=nn.CrossEntropyLoss(),
        batch_size=128,
        augmentation=True,
        dataset="cifar10",
        data_path=DATA_PATH,
        weights_folder=WEIGHTS_FOLDER,
        device=DEVICE,
        save=True
):
    trainloader = get_dataloader(
        dataset=dataset,
        root=data_path,
        train=True,
        train_batch_size=batch_size,
        augmentation=augmentation
    )
    testloader = get_dataloader(dataset=dataset, root=data_path, augmentation=augmentation)
    set_seed(42)
    pgd = PGD(
        eps=eps,
        step_size=step_size,
        mean=mean,
        std=std,
        targeted=targeted,
        batch_size=batch_size,
        device=device
    )
    model.to(device)

    for ep in range(epochs):
        train_loss_clean = 0.
        train_correct_clean = 0
        train_loss_adv = 0.
        train_correct_adv = 0
        train_total = 0

        model.train()
        with tqdm(trainloader, desc=f"{ep + 1}/{epochs} epochs:") as t:
            for i, (x, y) in enumerate(t):
                model.eval()
                with torch.no_grad():
                    out = model(x.to(device))
                    pred = out.max(dim=1)[1]
                    loss = loss_fn(out, y.to(device))
                train_loss_clean += loss.item() * x.size(0)
                train_correct_clean += (pred == y.to(device)).sum().item()
                train_total += x.size(0)
                x_adv = pgd.generate(model, x, y)
                model.train()
                out = model(x_adv.to(device))
                loss = loss_fn(out, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_adv += loss.item() * x.size(0)
                train_correct_adv += (pred == y.to(device)).sum().item()

                if i < len(t) - 1:
                    t.set_postfix({
                        "train_loss_clean": train_loss_clean / train_total,
                        "train_acc_clean": train_correct_clean / train_total,
                        "train_loss_adv": train_loss_adv / train_total,
                        "train_acc_adv": train_correct_adv / train_total,
                    })
                else:
                    test_loss_clean = 0.
                    test_correct_clean = 0
                    test_loss_adv = 0
                    test_correct_adv = 0
                    test_total = 0
                    model.eval()
                    for x, y in testloader:
                        with torch.no_grad():
                            out = model(x.to(device))
                            pred = out.max(dim=1)[1]
                            loss = loss_fn(out, y.to(device))
                        test_loss_clean += loss.item() * x.size(0)
                        test_correct_clean += (pred == y.to(device)).sum().item()
                        test_total += x.size(0)
                        x_adv = pgd.generate(model, x, y)
                        with torch.no_grad():
                            out = model(x_adv.to(device))
                            pred = out.max(dim=1)[1]
                            loss = loss_fn(out, y.to(device))
                        test_loss_adv += loss.item() * x.size(0)
                        test_correct_adv += (pred == y.to(device)).sum().item()
                    t.set_postfix({
                        "train_loss_clean": train_loss_clean / train_total,
                        "train_acc_clean": train_correct_clean / train_total,
                        "train_loss_adv": train_loss_adv / train_total,
                        "train_acc_adv": train_correct_adv / train_total,
                        "test_loss_clean": test_loss_clean / test_total,
                        "test_acc_clean": test_correct_clean / test_total,
                        "test_loss_adv": test_loss_adv / test_total,
                        "test_acc_adv": test_correct_adv / test_total,
                    })
        if scheduler is not None:
            scheduler.step()
    if save:
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
        model_name = model.__module__.split(".")[-1]
        torch.save(
            model.state_dict(),
            os.path.join(
                weights_folder,
                "_".join([dataset, model_name, "adv"]) + ".pt"
            ))
    return model


if __name__ == "__main__":
    from advkit.convnets.vgg import VGG
    from torch.optim import SGD

    model = VGG.from_default_config("vgg16")
    model.to(DEVICE)
    epochs = 1
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    adv_train(model, optimizer=optimizer, epochs=epochs)
