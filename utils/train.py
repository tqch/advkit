import torch
from tqdm import tqdm
import random
import numpy as np


def set_seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def train(model, epochs, trainloader, loss_fn, optim, valloader=None, device=torch.device("cpu")):

    # track the best epoch that has highest validation accuracy
    best_epoch, best_val_acc = None, None
    if valloader is not None:
        best_epoch = -1
        best_val_acc = 0

    for ep in range(epochs):
        train_loss = 0.
        train_correct = 0
        train_total = 0
        model.train()
        with tqdm(trainloader, desc=f"{ep + 1}/{epochs} epochs:") as t:
            for i, (x, y) in enumerate(t):
                optim.zero_grad()
                out = model(x.to(device))
                pred = out.max(dim=1)[1]
                loss = loss_fn(out, y.to(device))
                loss.backward()
                optim.step()
                train_loss += loss.item() * x.size(0)
                train_correct += (pred == y.to(device)).sum().item()
                train_total += x.size(0)
                if i < len(t) - 1 or valloader is None:
                    t.set_postfix(
                        {"train_loss": train_loss / train_total, "train_acc": train_correct / train_total})
                else:
                    val_loss = 0.
                    val_correct = 0
                    val_total = 0
                    model.eval()
                    with torch.no_grad():
                        for x, y in valloader:
                            out = model(x.to(device))
                            pred = out.max(dim=1)[1]
                            loss = loss_fn(out, y.to(device))
                            val_loss += loss.item() * x.size(0)
                            val_correct += (pred == y.to(device)).sum().item()
                            val_total += x.size(0)
                    t.set_postfix({"train_loss": train_loss / train_total, "train_acc": train_correct / train_total,
                                   "val_loss": val_loss / val_total, "val_acc": val_correct / val_total
                                   })
                    if val_correct / val_total > best_val_acc:
                        best_val_acc = val_correct / val_total
                        best_epoch = ep + 1

    return best_epoch, best_val_acc
