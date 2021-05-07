import os, time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from advkit.attacks.pgd import PGD
from collections import OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepFeatureDataset(Dataset):
    def __init__(self, data, deep_feature, label, knn_proba):
        self.data = data
        self.deep_feature = deep_feature
        self.label = label
        self.knn_proba = knn_proba
        self.size = len(self.knn_proba)

    def __getitem__(self, ind):
        return self.data[ind], self.deep_feature[ind], self.label[ind], self.knn_proba[ind]

    def __len__(self):
        return self.size


class AdvkNN:
    def __init__(
            self,
            in_features,
            out_features,
            model,
            train_data,
            train_targets,
            test_data=None,
            test_targets=None,
            hidden_layer=3,
            n_neighbors=5,
            metric="euclidean",
            batch_size=128,
            device=DEVICE,
            loss_fn=nn.KLDivLoss(reduction="batchmean")
    ):
        encoder = nn.Sequential()
        for i in range(hidden_layer + 2):
            module_name = "feature" if not i else f"layer{i}"
            try:
                encoder.add_module(
                    module_name, getattr(model, module_name)
                )
            except AttributeError:
                continue
        self._model = nn.Sequential(OrderedDict({
            "encoder": encoder,
            "flatten": nn.Flatten(start_dim=1),
            "dknnb": nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LogSoftmax(dim=1)
            )
        }))
        self._freeze_model()  # freeze the layers before the DkNN block
        self._model.eval()
        self._model.to(device)
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = loss_fn

        start = time.perf_counter()
        self.data = train_data
        self.deep_feature = self._extract_feature(train_data)
        self.train_targets = train_targets
        end = time.perf_counter()
        print("Feature extraction for training data takes {}s.".format(end - start))

        start = time.perf_counter()
        self._nn_clf = self._build_nn_clf()
        end = time.perf_counter()
        print("Building knn classifier for training data takes {}s.".format(end - start))

        start = time.perf_counter()
        self.knn_proba = self._get_knn_proba(self.deep_feature)
        end = time.perf_counter()
        print("KNN searching for training data takes {}s.".format(end - start))

        self.trainloader = DataLoader(DeepFeatureDataset(
            self.data, self.deep_feature, train_targets, self.knn_proba
        ), batch_size=self.batch_size, shuffle=True)
        if test_data is not None and test_targets is not None:
            start = time.perf_counter()
            test_deep_feature = self._extract_feature(test_data)
            self.testloader = DataLoader(DeepFeatureDataset(
                test_data, test_deep_feature, test_targets, self._get_knn_proba(test_deep_feature)
            ), batch_size=self.batch_size, shuffle=False)
            end = time.perf_counter()
            print("Building dataloader for test data takes {}s.".format(end - start))
        else:
            self.testloader = None

    def _extract_feature(self, data):
        dp_feat = []
        for i in range(0, len(data), self.batch_size):
            x_batch = data[i:i + self.batch_size]
            dp_feat_batch = self._model.encoder(x_batch.to(self.device)).detach().cpu().flatten(start_dim=1)
            if self.metric == "cosine":
                dp_feat_batch /= dp_feat_batch.pow(2).sum(dim=1,keepdim=True).sqrt()
            dp_feat.append(dp_feat_batch)
        dp_feat = torch.cat(dp_feat, dim=0)
        return dp_feat

    def _build_nn_clf(self):
        return KNeighborsClassifier(
            n_neighbors=self.n_neighbors, n_jobs=-1
        ).fit(self.deep_feature, self.train_targets)

    def _get_knn_proba(self, deep_feature):
        return self._nn_clf.predict_proba(deep_feature).astype("float32")

    def _freeze_model(self):
        for n, p in self._model.named_parameters():
            if "dknnb" not in n:
                p.requires_grad_(False)

    def train(
            self,
            epochs=20,
            num_eval_batches=4,
            **opt_params
    ):
        optimizer = SGD(
            getattr(self._model, "dknnb").parameters(),
            **opt_params
        )
        num_eval_batches = self.testloader is not None and (num_eval_batches or len(self.testloader))
        for e in range(epochs):
            running_loss = 0
            running_match = 0
            running_total = 0
            with tqdm(self.trainloader, desc=f"{e + 1}/{epochs} epochs") as t:
                for i, (_, dp_feat, _, p) in enumerate(t):
                    out = self._model.dknnb(dp_feat.to(self.device))
                    loss = self.loss_fn(out, p.to(self.device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    out_pred = out.max(dim=1)[1]
                    knn_pred = p.max(dim=1)[1]
                    running_match += (out_pred.cpu() == knn_pred).sum().item()
                    running_total += dp_feat.size(0)
                    if i == len(t) - 1 and self.testloader is not None:
                        test_running_loss = 0
                        test_running_match = 0
                        test_running_total = 0
                        for _, (_, dp_feat, _, p) in zip(range(num_eval_batches), self.testloader):
                            with torch.no_grad():
                                out = self._model.dknnb(dp_feat.to(self.device))
                            loss = self.loss_fn(out, p.to(self.device))
                            test_running_loss += loss.item()
                            out_pred = out.max(dim=1)[1]
                            knn_pred = p.max(dim=1)[1]
                            test_running_match += (out_pred.cpu() == knn_pred).sum().item()
                            test_running_total += dp_feat.size(0)
                        t.set_postfix({
                            "train_loss": running_loss / running_total,
                            "train_match_knn": running_match / running_total,
                            "test_loss": test_running_loss / test_running_total,
                            "test_match_knn": test_running_match / test_running_total
                        })
                    else:
                        t.set_postfix({
                            "train_loss": running_loss / running_total,
                            "train_match_knn": running_match / running_total
                        })

    def eval(
            self,
            num_test_batches=None,
            **attack_params
    ):
        pgd = PGD(
            **attack_params,
            loss_fn=self.loss_fn,
            batch_size=self.batch_size,
            device=self.device
        )

        test_correct_clean = 0
        test_correct_adv = 0
        test_total = 0

        num_test_batches = self.testloader is not None and (num_test_batches or len(self.testloader))

        for _, (x, _, y, p) in zip(range(num_test_batches), self.testloader):
            deep_feature_clean = self._extract_feature(x.to(self.device))
            pred_dknn_clean = self._get_knn_proba(deep_feature_clean).argmax(axis=1)
            test_correct_clean += (pred_dknn_clean == y.numpy()).sum()
            x_adv = pgd.generate(self._model, x, p)
            deep_feature_adv = self._extract_feature(x_adv.to(self.device))
            pred_dknn_adv = self._get_knn_proba(deep_feature_adv).argmax(axis=1)
            test_correct_adv += (pred_dknn_adv == y.numpy()).sum()
            test_total += x.size(0)
        print(f"The test clean accuracy of DkNN against is {test_correct_clean/test_total}")
        print(f"The test adversarial accuracy of DkNN against AdvkNN attack is {test_correct_adv/test_total}")


if __name__ == "__main__":
    from advkit.convnets.vgg import VGG

    optimizer_params = {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0002,
        "nesterov": True
    }

    ROOT = os.path.expanduser("~/advkit")
    DATA_PATH = os.path.join(ROOT, "datasets")
    WEIGHTS_PATH = os.path.join(ROOT, "model_weights/cifar10_vgg16.pt")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DOWNLOAD = not os.path.exists(os.path.join(DATA_PATH, "cifar-10-python.tar.gz"))

    batch_size = 128
    n_neighbors = 5

    trainset = datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=DOWNLOAD,
        transform=transforms.ToTensor()
    )
    testset = datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        download=DOWNLOAD,
        transform=transforms.ToTensor()
    )
    train_data, train_targets = (
        torch.FloatTensor(trainset.data.transpose(0, 3, 1, 2) / 255.)[:2000],
        torch.LongTensor(trainset.targets)[:2000]
    )
    test_data, test_targets = (
        torch.FloatTensor(testset.data.transpose(0, 3, 1, 2) / 255.)[:256],
        torch.LongTensor(testset.targets)[:256]
    )

    model = VGG.from_default_config("vgg16")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    in_features_layer = [16384, 8192, 4096, 2048, 512]
    hidden_layer = 3
    epochs = 30

    advknn = AdvkNN(
        in_features_layer[hidden_layer],
        10,
        model,
        train_data,
        train_targets,
        test_data=test_data,
        test_targets=test_targets,
        hidden_layer=hidden_layer
    )

    advknn.train(
        epochs=epochs,
        num_eval_batches=4,
        **optimizer_params
    )

    advknn.eval(
        eps=8/255,
        step_size=2/255,
        max_iter=10,
        random_init=True
    )