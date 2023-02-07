# MultivariGaussian_CL
Code for paper 'Gradient Regularization with Multivariate Distri-bution of Previous Knowledge for Continual Learning', IDEAL2022

written by Taeheon Kim, Hyungjun Moon, Sungbae Cho

Link: https://link.springer.com/book/10.1007/978-3-031-21753-1 page 359~368

-----------------------------------------------------------------------------------------------------------------------------------
# Abstract
Continual learning is a novel learning setup for an environment where data are introduced sequentially, and a model continually learns new tasks. However, the model forgets the learned knowledge as it learns new classes. There is an approach that keeps a few previous data, but this causes other problems such as
overfitting and class imbalance. In this paper, we propose a method that retrains a network with generated representations from an estimated multivariate Gaussian distribution. The representations are the vectors coming from CNN that is trained using a gradient regularization to prevent a distribution shift, allowing the
stored means and covariances to create realistic representations. The generated vectors contain every class seen so far, which helps preventing the forgetting.
Our 6-fold cross-validation experiment shows that the proposed method outperforms the existing continual learning methods by 1.14%p and 4.60%p in CIFAR10 and CIFAR100, respectively. Moreover, we visualize the generated vectors using t-SNE to confirm the validity of multivariate Gaussian mixture to estimate the distribution of the data representations.


# Overview
![image](https://user-images.githubusercontent.com/99951369/217221449-e3d168e6-3feb-4a27-b5f2-9a1679d2d64a.png)
