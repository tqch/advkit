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
            metric="euclidean",
            batch_size=128,
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
        self.metric = metric
        self.batch_size = batch_size
        self._nns = self._build_nns()
        self.n_class = n_class

    @staticmethod
    def _split_data(train_data, train_targets):
        return train_data, None, train_targets, None

    def _get_hidden_repr(self, x, return_targets=False):

        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i:i + self.batch_size]
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

        if self.metric == "cosine":
            hidden_reprs = [
                hidden_repr / np.sqrt(np.power(hidden_repr, 2).sum(axis=1, keepdims=True))
                for hidden_repr in hidden_reprs
            ]

        if return_targets:
            outs = np.concatenate(outs, axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs, targets

    def _wrap_model(self, model):

        class ModelWrapper(nn.Module):

            def __init__(self, model, hidden_layers):
                super(ModelWrapper, self).__init__()
                self._model = model
                self.feature = self._model.feature \
                    if hasattr(self._model, "feature") else nn.Identity()
                self.intermediate_layers = [
                    mod[1]
                    for mod in model.named_children()
                    if not ("feature" in mod[0] or "classifier" in mod[0])
                ]
                self.classifier = self._model.classifier\
                    if hasattr(self._model, "classifier") else nn.Identity()
                if hidden_layers == -1:
                    self.hidden_layers = tuple(range(len(self.intermediate_layers)))
                else:
                    self.hidden_layers = hidden_layers

            def forward(self, x):
                hidden_reprs = []
                x = self.feature(x)
                for block in self.intermediate_layers:
                    x = block(x)
                    hidden_reprs.append(x.detach().cpu())
                try:
                    out = self.classifier(x).detach().cpu()
                except RuntimeError:
                    out = None
                return [hidden_reprs[i] for i in self.hidden_layers], out

            def forward_branch(self, hidden_layer):

                mappings = self.intermediate_layers[:hidden_layer + 1]

                def branch(x):
                    out = self.feature(x)
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

    def _get_nns(self, x):
        hidden_reprs, _ = self._get_hidden_repr(x)
        knns = [nn.kneighbors(hidden_repr, return_distance=False)
                for hidden_repr, nn in zip(hidden_reprs, self._nns)]
        knns = np.concatenate(knns, axis=1)
        return knns

    def predict_proba(self, knn_labs):
        return np.stack(list(map(
            lambda x: np.bincount(x, minlength=10),
            knn_labs
        ))) / knn_labs.shape[1]


class DkNN(DkNNBase):

    def __init__(
            self,
            model,
            train_data,
            train_targets,
            calib_size=0.02,
            n_class=10,
            hidden_layers=-1,
            n_neighbors=5,
            metric="euclidean",
            batch_size=128,
            device=torch.device("cpu")
    ):
        self.calib_size = calib_size
        super(DkNN, self).__init__(
            model,
            train_data,
            train_targets,
            n_class,
            hidden_layers,
            n_neighbors=n_neighbors,
            metric=metric,
            batch_size=batch_size,
            device=device
        )
        self._calib_alphas = self._calc_alpha(calibration=True)

    def _split_data(
            self,
            train_data,
            train_targets,
            calib_size=0.02,
            random_state=42
    ):
        train_inds, calib_inds = train_test_split(
            np.arange(len(train_targets)),
            shuffle=True,
            test_size=self.calib_size,
            random_state=random_state
        )
        train_inds = torch.LongTensor(train_inds)
        calib_inds = torch.LongTensor(calib_inds)
        return (
            train_data[train_inds], train_data[calib_inds],
            train_targets[train_inds], train_targets[calib_inds]
        )

    def _calc_alpha(self, x=None, calibration=False):

        knns = self._get_nns(self.calib_data) if calibration else self._get_nns(x)
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


class SimDkNN(DkNNBase):

    def __init__(
            self,
            model,
            train_data,
            train_targets,
            n_class=10,
            hidden_layers=-1,
            n_neighbors=5,
            metric="euclidean",
            batch_size=128,
            device=torch.device("cpu")
    ):
        super(SimDkNN, self).__init__(
            model,
            train_data,
            train_targets,
            n_class,
            hidden_layers,
            n_neighbors=n_neighbors,
            metric=metric,
            batch_size=batch_size,
            device=device
        )

    def __call__(self, x):
        knns = self._get_nns(x)
        knn_labs = self.train_targets[knns]
        return self.predict_proba(knn_labs)


if __name__ == "__main__":
    import os
    from advkit.attacks.pgd import PGD
    from advkit.utils.data import get_dataloader, DATA_PATH, CHECKPOINT_FOLDER
    from advkit.utils.models import get_model
    from torchvision.datasets import CIFAR10

    CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, "cifar10_vgg16.pt")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DOWNLOAD = not os.path.exists(os.path.join(DATA_PATH, "cifar-10-python.tar.gz"))

    trainset = CIFAR10(root=DATA_PATH, train=True, download=DOWNLOAD)
    train_data, train_targets = (
        torch.FloatTensor(trainset.data.transpose(0, 3, 1, 2) / 255.)[:2000],
        torch.LongTensor(trainset.targets)[:2000]
    )  # for memory's sake, only take 2000 as train set
    testloader = get_dataloader(dataset="cifar10", test_batch_size=128)

    model = get_model("vgg", "vgg16")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model"])
    model.eval()
    model.to(DEVICE)

    dknn = SimDkNN(
        model,
        train_data,
        train_targets,
        hidden_layers=[3, ],
        metric="cosine",
        device=DEVICE
    )

    pgd = PGD(
        eps=8 / 255.,
        step_size=2 / 255.,
        batch_size=128,
        device=DEVICE
    )
    x, y = next(iter(testloader))
    x_adv = pgd.generate(model, x, y)
    pred_benign = dknn(x.to(DEVICE)).argmax(axis=1)
    acc_benign = (pred_benign == y.numpy()).astype("float").mean()
    print(f"The benign accuracy is {acc_benign}")

    pred_adv = dknn(x_adv.to(DEVICE)).argmax(axis=1)
    acc_adv = (pred_adv == y.numpy()).astype("float").mean()
    print(f"The adversarial accuracy is {acc_adv}")
