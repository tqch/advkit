# Adversarial Learning Kit

## Introduction

`advkit` package is a byproduct of my adversarial learning research project at University of Michigan, **A Robust Adversarial Immune-inspired Learning System [(RAILS)](https://arxiv.org/abs/2012.10485)**. Currently, the package is still under construction. There are three basic modules, `.convnet` (convolutional neual network backbones), `attacks` (attack strategies) and `.denfenses` (defense strategies or models).


## Contents of modules and references:

### ConvNets:

- **VGG:** Very Deep Convolutional Networks for Large-Scale Image Recognition [[Simonyan et al., 2014]](https://arxiv.org/abs/1409.1556)
- **ResNet:** Deep Residual Learning for Image Recognition [[He et al., 2015]](https://arxiv.org/abs/1512.03385)
- **(Not Implemented) InceptionNet:** [[Szegedy et al., 2015]](https://arxiv.org/abs/1512.00567)
- **DenseNet:** Densely Connected Convolutional Networks [[Huang et al., 2016]](https://arxiv.org/abs/1608.06993)

### Attacks

- **PGD:** Towards Deep Learning Models Resistant to Adversarial Attacks [[Madry et al., 2017]](https://arxiv.org/abs/1706.06083)
- **DkNN Attack:** On the Robustness of Deep K-Nearest Neighbors [[Sitawarin and Wagner, 2019]](https://arxiv.org/abs/1903.08333)
- **(Not implemented) Adv-kNN:** Adversarial Attacks On K-Nearest Neighbor Classifiers With Approximate Gradients [[Li et al., 2019]](https://arxiv.org/abs/1911.06591) 

### Defenses

- **Adversarial training:** Towards Deep Learning Models Resistant to Adversarial Attacks [[Madry et al., 2017]](https://arxiv.org/abs/1706.06083)
- **DkNN:** Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning [[Papernot et al., 2018]](https://arxiv.org/abs/1803.04765)
- **TRADES:** Theoretically Principled Trade-off between Robustness and Accuracy [[Zhang et al., 2019]](https://arxiv.org/abs/1901.08573)
  
*Note: TRADES is the abbreviation of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization*

