import os
import torch
from tqdm import tqdm
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR, LambdaLR
import advkit.convnets as cns

OPTIMIZER_CONFIGS = {
    "mnist": {
        "lr": 0.01
    },
    "cifar10": {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "nesterov": True
    }
}

MODEL_DICT = {
    "vgg": cns.vgg.VGG,
    "resnet": cns.resnet.ResNet,
    "wide_resnet": cns.wide_resnet.WideResNet,
    "densenet": cns.densenet.DenseNet
}


def set_seed(seed=71):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
        model,
        epochs,
        trainloader,
        loss_fn,
        optimizer,
        scheduler=None,
        valloader=None,
        num_eval_batches=-1,
        checkpoint_path=None,
        device=torch.device("cpu")
):

    n_iter = len(trainloader)  # number of iterations per epoch

    if checkpoint_path is not None:
        print(os.path.exists(checkpoint_path))
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint.keys() == {"epoch", "val_acc", "model", "optimizer_state"}, \
                "invalid checkpoint"
            best_epoch = checkpoint["epoch"] - 1
            best_val_acc = checkpoint["val_acc"]
            train_epochs = epochs - checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None:
                if isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step(best_epoch + 2 - 1 / n_iter)
                elif isinstance(scheduler, (StepLR, LambdaLR)):
                    scheduler.step(best_epoch)

    else:
        train_epochs = epochs
        # track the best epoch that has highest validation accuracy
        best_epoch, best_val_acc = -1, 0

    if valloader is not None:
        assert num_eval_batches < n_iter, \
            "number of evaluation batches should be less than the number of iterations per epoch"
        if num_eval_batches == -1:
            num_eval_batches = n_iter

    for ep in range(train_epochs):
        train_loss = 0.
        train_correct = 0
        train_total = 0
        model.train()
        with tqdm(trainloader, desc=f"{ep + best_epoch + 2}/{epochs} epochs:") as t:
            for i, (x, y) in enumerate(t):
                out = model(x.to(device))
                pred = out.max(dim=1)[1]
                loss = loss_fn(out, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if isinstance(optimizer, CosineAnnealingWarmRestarts):
                    scheduler.step(ep + best_epoch + 1 + i / n_iter)
                train_loss += loss.item() * x.size(0)
                train_correct += (pred == y.to(device)).sum().item()
                train_total += x.size(0)
                if i < len(t) - 1 or valloader is None:
                    t.set_postfix({
                        "train_loss": train_loss / train_total,
                        "train_acc": train_correct / train_total
                    })
                else:
                    val_loss = 0.
                    val_correct = 0
                    val_total = 0
                    model.eval()
                    with torch.no_grad():
                        for _, (x, y) in zip(range(num_eval_batches), valloader):
                            out = model(x.to(device))
                            pred = out.max(dim=1)[1]
                            loss = loss_fn(out, y.to(device))
                            val_loss += loss.item() * x.size(0)
                            val_correct += (pred == y.to(device)).sum().item()
                            val_total += x.size(0)
                    t.set_postfix({
                        "train_loss": train_loss / train_total,
                        "train_acc": train_correct / train_total,
                        "val_loss": val_loss / val_total,
                        "val_acc": val_correct / val_total
                    })
                    if val_correct / val_total > best_val_acc:
                        best_val_acc = val_correct / val_total
                        best_epoch = ep + best_epoch + 2
                        if checkpoint_path is not None:
                            if not os.path.exists(os.path.dirname(checkpoint_path)):
                                os.makedirs(os.path.dirname(checkpoint_path))
                            torch.save({
                                "epoch": best_epoch,
                                "val_acc": best_val_acc,
                                "model_weights": model.state_dict(),
                                "optimizer_state": optimizer.state_dict()
                            }, checkpoint_path)
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_correct / val_total)
            if isinstance(scheduler, (StepLR, LambdaLR)):
                scheduler.step()

    return best_epoch, best_val_acc


def evaluate(
        model,
        testloader,
        device=torch.device("cpu")
):
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for x, y in testloader:
            out = model(x.to(device))
            pred = out.max(dim=1)[1]
            test_correct += (pred == y.to(device)).sum().item()
            test_total += x.size(0)
    print("Test accuracy is %.3f" % (test_correct / test_total))


def get_model(
        model_type,
        default_config=None,
        **model_params,
):
    model = MODEL_DICT[model_type]
    if default_config is None:
        return model(**model_params)
    else:
        return model.from_default_config(default_config)
