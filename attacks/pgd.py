import torch
import torch.nn as nn


class PGD:
    def __init__(
            self,
            eps=60 / 255.,
            step_size=20 / 255.,
            max_iter=20,
            random_init=False,
            targeted=False,
            loss_fn=nn.CrossEntropyLoss(),
            batch_size=64
    ):
        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter
        self.random_init = random_init
        self.targeted = targeted
        self.loss_fn = loss_fn
        self.batch_size = batch_size

    def attack(
            self,
            model,
            x,
            y,
            x_adv=None,
            targets=None
    ):
        if x_adv is None:
            if self.random_init:
                x_adv = self.step_size * torch.randn_like(x) + x
                x_adv = x_adv.clamp(0.0, 1.0)
            else:
                x_adv = torch.clone(x).detach()
        x_adv.requires_grad_(True)
        pred_adv = model(x_adv)
        if self.targeted:
            assert targets is not None, "Target labels not found!"
            loss = self.loss_fn(pred_adv, targets)
        else:
            loss = self.loss_fn(pred_adv, y)
        loss.backward()
        pert = self.step_size * x_adv.grad.sign()
        x_adv = (x_adv + pert).clamp(0.0, 1.0).detach()
        pert = (x_adv - x).clamp(-self.eps, self.eps)
        return x + pert

    def generate(
            self,
            model,
            x,
            y=None,
            targets=None,
            device=torch.device("cpu")
    ):
        model.to(device)
        model.eval()  # ensure that the model is run in the evaluation mode
        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i + self.batch_size].to(device)
            if y is None:
                y_batch = model(x_batch)
                if isinstance(y_batch, (list, tuple)):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(device)
            else:
                y_batch = y[i: i + self.batch_size].to(device)
            for j in range(self.max_iter):
                if j == 0:
                    x_adv_batch = self.attack(model, x_batch, y_batch, targets=targets)
                else:
                    x_adv_batch = self.attack(model, x_batch, y_batch, x_adv_batch, targets=targets)
            x_adv.append(x_adv_batch)
        return torch.cat(x_adv, dim=0).cpu()


if __name__ == "__main__":
    from utils.data import get_dataloader
    from convnets.vgg import VGG16

    ROOT = "../datasets"
    MODEL_WEIGHTS = "../model_weights/cifar_vgg16.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset = get_dataloader(dataset="cifar10", root=ROOT)
    model = VGG16()
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    pgd = PGD(eps=8 / 255., step_size=2 / 255, batch_size=128)
    x, y = next(iter(testset))
    x_adv = pgd.generate(model, x, y, device=DEVICE)

    pred_adv = model(x_adv.to(DEVICE)).max(dim=1)[1].detach()
    acc_adv = (pred_adv == y.to(pred_adv)).float().mean().item()
    print(f"The adversarial accuracy is {acc_adv}")
