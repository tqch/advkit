import torch
from tqdm import tqdm
import random
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def set_seed(seed=71):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
        model,
        epochs,
        train_loader,
        loss_fn,
        optimizer,
        scheduler=None,
        val_loader=None,
        device=torch.device("cpu")
):

    # track the best epoch that has highest validation accuracy
    best_epoch, best_val_acc = None, None

    if val_loader is not None:
        best_epoch = -1
        best_val_acc = 0

    for ep in range(epochs):
        train_loss = 0.
        train_correct = 0
        train_total = 0
        model.train()
        with tqdm(train_loader, desc=f"{ep + 1}/{epochs} epochs:") as t:
            for i, (x, y) in enumerate(t):
                optimizer.zero_grad()
                out = model(x.to(device))
                pred = out.max(dim=1)[1]
                loss = loss_fn(out, y.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_correct += (pred == y.to(device)).sum().item()
                train_total += x.size(0)
                if i < len(t) - 1 or val_loader is None:
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
                        for x, y in val_loader:
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
                        best_epoch = ep + 1
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_correct/val_total)
            else:
                scheduler.step()

    return best_epoch, best_val_acc


def evaluate(
        model,
        test_loader,
        device=torch.device("cpu")
):
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for x, y in test_loader:
            out = model(x.to(device))
            pred = out.max(dim=1)[1]
            test_correct += (pred == y.to(device)).sum().item()
            test_total += x.size(0)
    print("Test accuracy is %.3f" % (test_correct / test_total))