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
        eps=8 / 255,
        step_size=2 / 255,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        targeted=False,
        loss_fn=nn.CrossEntropyLoss(),
        batch_size=128,
        num_eval_batches=None,
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
    testloader = get_dataloader(
        dataset=dataset,
        root=data_path,
        test_batch_size=batch_size,
        augmentation=augmentation
    )
    if num_eval_batches == -1:
        num_eval_batches = len(testloader)
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
                pred = out.max(dim=1)[1]
                loss = loss_fn(out, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_adv += loss.item() * x.size(0)
                train_correct_adv += (pred == y.to(device)).sum().item()

                if i < len(t) - 1 or num_eval_batches is None:
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
                    for _, (x, y) in zip(range(num_eval_batches), testloader):
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
    import math
    from advkit.convnets.vgg import VGG
    from advkit.utils.data import WEIGHTS_FOLDER, DATA_PATH, DATASET_CONFIGS
    from torch.optim import SGD, lr_scheduler

    model = VGG.from_default_config("vgg16")
    model.to(DEVICE)
    CHECKPOINT_PATH = os.path.join(WEIGHTS_FOLDER, "cifar10_vgg_adv.pt")
    augmentation = True
    mean, std = (0, 0, 0), (1, 1, 1)
    if augmentation:
        mean, std = DATASET_CONFIGS["cifar10"]["mean"], DATASET_CONFIGS["cifar10"]["std"]

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model"])
        model.eval()
        testloader = get_dataloader(
            "cifar10",
            root=DATA_PATH,
            test_batch_size=512,
            augmentation=augmentation
        )
        pgd = PGD(
            eps=8 / 255,
            step_size=10,
            mean=mean,
            std=std,
            device=DEVICE
        )
        test_correct_benign = 0
        test_correct_adv = 0
        test_total = 0
        for (x, y) in testloader:
            x_adv = pgd.generate(model, x, y)
            with torch.no_grad():
                pred_benign = model(x.to(DEVICE)).max(dim=1)[1]
                pred_adv = model(x_adv.to(DEVICE)).max(dim=1)[1]
            test_correct_benign += (pred_benign == y.to(DEVICE)).sum().item()
            test_correct_adv += (pred_adv == y.to(DEVICE)).sum().item()
            test_total += x.size(0)
        print(
            "The benign accuracy is %f\nThe adversarial accuracy against PGD attack is %f"
            % (test_correct_benign / test_total, test_correct_adv / test_total)
        )
    else:
        epochs = 200
        lr = 0.1
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: lr - 0.45 * lr * (1 - math.cos(math.pi * epoch / epochs))
        )
        adv_train(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            mean=mean,
            std=std,
            num_eval_batches=2,
            augmentation=augmentation
        )
