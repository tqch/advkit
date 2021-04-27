import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DkNNBase:

    def __init__(
            self,
            model,
            train_data,
            train_targets,
            n_class=10,
            hidden_layers=-1,
            n_neighbors=5,
            device=torch.device("cpu")
    ):

        self.hidden_layers = hidden_layers
        self._model = self._wrap_model(model)
        self.hidden_layers = self._model.hidden_layers
        self.device = device
        self._model.eval()  # make sure the model is in the eval mode
        self._model.to(self.device)

        self.train_data, self.calib_data, self.train_targets, self.calib_targets = \
            self._split_data(torch.FloatTensor(train_data), np.array(train_targets))

        self.n_neighbors = n_neighbors
        self._nns = self._build_nns()
        self.n_class = n_class

    @staticmethod
    def _split_data(train_data, train_targets):
        return train_data, None, train_targets, None

    def _get_hidden_repr(self, x, batch_size=256, return_targets=False):

        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), batch_size):
            x_batch = x[i:i + batch_size]
            if return_targets:
                hidden_reprs_batch, outs_batch = self._model(x_batch.to(self.device))
            else:
                hidden_reprs_batch, _ = self._model(x_batch.to(self.device))
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [np.concatenate([
            hidden_repr[i].flatten(start_dim=1)
            for hidden_repr in hidden_reprs
        ], axis=0) for i in range(len(self.hidden_layers))]

        if return_targets:
            outs = np.concatenate(outs, axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs, targets

    def _wrap_model(self, model):

        class ModelWrapper(nn.Module):

            def __init__(self, model, hidden_layers):
                super(ModelWrapper, self).__init__()
                self._model = model
                self.initial_block = self._model.initial_block\
                    if hasattr(self._model, "initial_block") else nn.Identity()
                self.intermediate_blocks = [
                    mod[1] for mod in model.named_children() if not ("initial" in mod[0] or "final" in mod[0])
                ]
                self.final_block = self._model.final_block if hasattr(self._model, "final_block") else nn.Identity()
                if hidden_layers == -1:
                    self.hidden_layers = tuple(range(len(self.intermediate_blocks)))
                else:
                    self.hidden_layers = hidden_layers

            def forward(self, x):
                hidden_reprs = []
                x = self.initial_block(x)
                for block in self.intermediate_blocks:
                    x = block(x)
                    hidden_reprs.append(x.detach().cpu())
                out = self.final_block(x).detach().cpu()
                return [hidden_reprs[i] for i in self.hidden_layers], out

            def forward_branch(self, hidden_layer):

                mappings = self.intermediate_blocks[:hidden_layer+1]

                def branch(x):
                    out = self.initial_block(x)
                    for mp in mappings:
                        out = mp(out)
                    return out.flatten(start_dim=1)

                return branch

        return ModelWrapper(model, self.hidden_layers)

    def _build_nns(self):
        hidden_reprs, _ = self._get_hidden_repr(self.train_data, return_targets=False)

        return [
            NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(hidden_repr)
            for hidden_repr in tqdm(hidden_reprs, desc="Building nearest neighbors classifiers")
        ]


class DkNN(DkNNBase):

    def __init__(
            self,
            model,
            train_data,
            train_targets,
            n_class=10,
            hidden_layers=-1,
            n_neighbors=5,
            device=torch.device("cpu")
    ):

        super(DkNN, self).__init__(model, train_data, train_targets, n_class, hidden_layers, n_neighbors, device)
        self._calib_alphas = self._calc_alpha(calibration=True)

    @staticmethod
    def _split_data(
            train_data,
            train_targets,
            calib_size=0.02,
            random_state=42
    ):
        train_inds, calib_inds = train_test_split(
            np.arange(len(train_targets)),
            shuffle=True,
            test_size=calib_size,
            random_state=random_state
        )
        train_inds = torch.LongTensor(train_inds)
        calib_inds = torch.LongTensor(calib_inds)
        return (
            train_data[train_inds], train_data[calib_inds],
            train_targets[train_inds], train_targets[calib_inds]
        )

    def _calc_alpha(self, x=None, calibration=False):

        if not calibration:
            hidden_reprs, _ = self._get_hidden_repr(x)
        else:
            hidden_reprs, _ = self._get_hidden_repr(self.calib_data)

        knns = [nn.kneighbors(hidden_repr, return_distance=False)
                for hidden_repr, nn in zip(hidden_reprs, self._nns)]
        knns = np.concatenate(knns, axis=1)
        knn_labs = self.train_targets[knns]

        if not calibration:
            alpha = [(knn_labs != i).sum(axis=1) for i in range(self.n_class)]
            alpha = np.stack(alpha, axis=0).T
        else:
            alpha = (knn_labs != self.calib_targets[:, None]).sum(axis=1)
            alpha = np.bincount(alpha, minlength=self.n_neighbors * len(self.hidden_layers) + 1)
            alpha = np.cumsum(alpha)

        return alpha

    def __call__(self, x):
        alpha = self._calc_alpha(x)
        pvalue = 1 - self._calib_alphas[alpha] / len(self.calib_targets)
        prediction = pvalue.argmax(axis=1)
        confidence = pvalue.max(axis=1)
        credibility = np.sort(pvalue, axis=1)[:, -2]
        return prediction, confidence, credibility


if __name__ == "__main__":
    import os
    from attacks.pgd import PGD
    from utils.data import get_dataloader
    from convnets.vgg import VGG
    from torchvision.datasets import CIFAR10

    ROOT = "../datasets"
    MODEL_WEIGHTS = "../model_weights/cifar_vgg16.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DOWNLOAD = not os.path.exists("../datasets/cifar-10-python.tar.gz")

    trainset = CIFAR10(root=ROOT, train=True, download=DOWNLOAD)
    train_data, train_targets = (
        torch.FloatTensor(trainset.data.transpose(0, 3, 1, 2) / 255.)[:2000],
        torch.LongTensor(trainset.targets)[:2000]
    )  # for memory's sake, only take 2000 as train set

    test_loader = get_dataloader(dataset="cifar10", root=ROOT, test_batch_size=128)

    model = VGG.from_default_config("vgg16")
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    dknn = DkNN(model, train_data, train_targets, device=DEVICE)

    pgd = PGD(eps=8 / 255., step_size=2 / 255., batch_size=128)
    x, y = next(iter(test_loader))
    x_adv = pgd.generate(model, x, y, device=DEVICE)

    pred_benign, _, _ = dknn(x.to(DEVICE))
    acc_benign = (pred_benign == y.numpy()).astype("float").mean()
    print(f"The benign accuracy is {acc_benign}")

    pred_adv, _, _ = dknn(x_adv.to(DEVICE))
    acc_adv = (pred_adv == y.numpy()).astype("float").mean()
    print(f"The adversarial accuracy is {acc_adv}")
