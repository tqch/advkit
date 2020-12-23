import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import numpy as np


class DKNN:

    def __init__(self,model,train_data,train_targets,n_class=10,hidden_layers=-1,n_neighbors=5,device=torch.device("cpu")):

        self.hidden_layers = hidden_layers
        self._model = self.wrap_model(model)
        self.hidden_layers = self._model.hidden_layers
        self.device = device
        self._model.eval() # make sure the model is in the eval mode
        self._model.to(self.device)

        self.n_class = n_class

        self.train_data,self.calib_data,self.calib_data,self.calib_targets =\
            self._build_calib(train_data,train_targets)

        self.n_neighbors = n_neighbors
        self._nns = self._build_nns()

        self._calib_alphas = self._calc_alpha(calibration=True)

    @staticmethod
    def _build_calib(train_data,train_targets,calib_size=0.02,random_state=42):
        return train_test_split(
            train_data,
            train_targets,
            shuffle=True,
            test_size = calib_size,
            random_state=random_state
        )

    def _get_hidden_repr(self,x,batch_size=1024,return_targets=False):

        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0,x.size(0),batch_size):
            x_batch = x[i:i+batch_size]
            if return_targets:
                hidden_reprs_batch,outs_batch = self._model(x_batch.to(self.device))
            else:
                hidden_reprs_batch,_ = self._model(x_batch.to(self.device))
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [np.concatenate([hidden_repr[i] for hidden_repr in hidden_reprs],axis=0)\
                        for i in range(self.hidden_layers)]

        if return_targets:
            outs = np.concatenate(outs,axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs,targets

    def _wrap_model(self):

        class ModelWrapper(nn.Module):

            def __init__(self,model,hidden_layers):
                super(ModelWrapper,self).__init__()
                self._model = model
                self.blocks = [m for m in model.modules() if isinstance(nn.Sequential)]
                if hidden_layers == -1:
                    self.hidden_layers = len(self.blocks)
                else:
                    self.hidden_layers = hidden_layers
                self.final_layers = self._model.final_layer()

            def forward(self,x):
                hidden_reprs = []
                for block in self.blocks:
                    x = block(x)
                    hidden_reprs.append(x)
                out = self.final_layers(x).detach().cpu()
                hidden_reprs = [hidden_repr.detach().cpu() for hidden_repr in hidden_reprs]
                return hidden_reprs[self.hidden_layers],out

        return ModelWrapper(self.model,self.hidden_layers)

    def _build_nns(self):
        hidden_reprs,_ = self._get_hidden_repr(self.train_data,return_targets=False)
        return [
            NearestNeighbors(n_neighbors=self.n_neighbors,n_jobs=-1).fit(hidden_repr)
            for hidden_repr in hidden_reprs
        ]

    def _calc_alpha(self,x=None,calibration=False):

        if not calibration:
            hidden_reprs,_ = self._get_hidden_repr(x)
        else:
            hidden_reprs,_ = self._get_hidden_repr(self.calib_data)

        knns = [nn.kneighbors(hidden_repr,return_distance=False)
                      for hidden_repr,nn in zip(hidden_reprs,self.nns)]
        knns = np.concatenate(knns,axis=1)

        knn_labs = self.train_targets[knns]

        if not calibration:
            alphas = [(knn_labs!=i).sum(axis=1) for i in range(self.n_class)]
            alphas = np.hstack(alphas)
        else:
            alphas = (knn_labs!=self.calib_targets[:,None]).sum(axis=1)
            alphas = np.bincount(alphas,minlength=self.n_neighbors*len(self.hidden_layers)+1)
            alphas = np.cumsum(alphas)

        return alphas

    def _calc_pvalue(self,x):
        alpha = self._calc_alpha(x)
        pvalue = 1-self._calib_alphas[alphas]/self.n_neighbors*len(self.hidden_layers)
        prediction = pvalue.argmax(axis=1)
        confidence = pvalue.max(axis=1)
        credibility = np.sort(pvalue,axis=1)[:,-2]
        return prediction,confidence,credibility

if __name__ == "__main__":
    import os
    from torchvision import datasets,transforms

    root = "./datasets"
    download = os.path.exists(root)
    transform = os.path

    trainset = datasets.CIFAR10(root=root,download=download,train=True,transform=transform)
    testset = datasets.CIFAR10(root=root,download=download,train=False,transform=transform)

    # train_mean = [0.49139968, 0.48215841, 0.44653091]
    # train_std = [0.24703223, 0.24348513, 0.26158784]


