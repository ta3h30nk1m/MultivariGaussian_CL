from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utils import *

reg_x = torch.empty(0)
buffer = Buffer
d_k = own_MN(torch.Tensor(buffer[0].means_[0]).cuda(), torch.Tensor(buffer[0].covariances_[0]).cuda())
for _ in range(5):
    shape = d_k._extended_shape(torch.Size((200,)))
    eps = _standard_normal(shape, dtype=d_k.loc.dtype, device=d_k.loc.device)
    x = []
    for (mean, covariance) in zip(buffer[0].means_, buffer[0].covariances_):
        mean, covariance = torch.Tensor(mean).cuda(), torch.Tensor(covariance).cuda()
        d_k = own_MN(mean, covariance)
        x_k = d_k.rsample(eps)
        x.append(x_k)
    X = torch.max(x[0],x[1]).cpu()
    X[X<0]=0
    reg_x = torch.cat((reg_x, X))
x = np.concatenate((Features[0], reg_x.detach().numpy()))
y = np.concatenate((np.full(5000, 'real'), np.full(1000, 'generated')))

tsne = TSNE(n_components=2, random_state=42)
result = tsne.fit_transform(x)
df = pd.DataFrame()
df['label']= y
df['x'] = result[:,0]
df['y'] = result[:,1]
# colors={0:"red", 
#         1: "orange",
#         2:"yellow",
#         3:"lightgreen",
#         4:"aqua",
#         5:"navy",
#         6:"darkviolet",
#         7:"magenta",
#         8:"brown", 
#         9:"black", }
colors = {'real': 'lightgreen','generated':'red'}

plt.figure(figsize = (16,16))
sns.scatterplot(x='x', y='y', hue='label', palette=colors, alpha=0.5, data=df).set(title="title")