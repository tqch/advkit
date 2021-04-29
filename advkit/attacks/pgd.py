import torch
import torch.nn as nn


class PGD:
    def __init__(
            self,
            eps=8 / 255.,
            step_size=2 / 255.,
            mean=(0, 0, 0),
            std=(1, 1, 1),
            max_iter=20,
            random_init=True,
            targeted=False,
            loss_fn=nn.CrossEntropyLoss(),
            batch_size=64,
            device = torch.device("cpu")
    ):
        self.eps = eps
        self.step_size = step_size

        self.max_iter = max_iter
        self.random_init = random_init
        self.targeted = targeted
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.device = device

        self.mean = torch.tensor(mean)[None, :, None, None].to(self.device)
        self.std = torch.tensor(std)[None, :, None, None].to(self.device)

    def _clip(
            self,
            x,
            center=None,
            eps=(0, 1)
    ):
        if center is None:
            x_denormalized = x * self.std + self.mean
            x_clipped = (x_denormalized.clamp(*eps) - self.mean) / self.std
        else:
            x_clipped = ((x-center) * self.std).clamp(-eps, eps) / self.std + center

        return x_clipped

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
                x_adv = self.step_size / self.std * torch.randn_like(x) + x
                x_adv = self._clip(x_adv)
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
        pert = self.step_size / self.std * x_adv.grad.sign()
        x_adv = self._clip(x_adv.data + pert)
        pert = self._clip(x_adv, center=x, eps=self.eps) - x
        return x + pert

    def generate(
            self,
            model,
            x,
            y=None,
            targets=None
    ):
        model.to(self.device)
        model.eval()  # ensure that the model is run in the evaluation mode
        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i + self.batch_size].to(self.device)
            if y is None:
                y_batch = model(x_batch)
                if isinstance(y_batch, (list, tuple)):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(self.device)
            else:
                y_batch = y[i: i + self.batch_size].to(self.device)
            for j in range(self.max_iter):
                if j == 0:
                    x_adv_batch = self.attack(model, x_batch, y_batch, targets=targets)
                else:
                    x_adv_batch = self.attack(model, x_batch, y_batch, x_adv_batch, targets=targets)
            x_adv.append(x_adv_batch)
        return torch.cat(x_adv, dim=0).cpu()


if __name__ == "__main__":
    from advkit.utils.data import get_dataloader
    from advkit.convnets.vgg import VGG
    import os

    ROOT = os.path.expanduser("~/advkit")
    DATA_PATH = os.path.join(ROOT, "datasets")
    WEIGHTS_PATH = os.path.join(ROOT, "model_weights/cifar10_vgg16.pt")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset = get_dataloader(dataset="cifar10", root=DATA_PATH)
    model = VGG.from_default_config("vgg16")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    pgd = PGD(
        eps=8 / 255.,
        step_size=2 / 255,
        batch_size=128,
        device=DEVICE
    )
    x, y = next(iter(testset))
    x_adv = pgd.generate(model, x, y)

    pred_adv = model(x_adv.to(DEVICE)).max(dim=1)[1].detach()
    acc_adv = (pred_adv == y.to(pred_adv)).float().mean().item()
    print(f"The adversarial accuracy is {acc_adv}")
