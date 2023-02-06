import numpy as np
import torch
from sklearn.utils import shuffle
from PIL import Image

class CLDataLoader(object):
    def __init__(self, datasets_per_task, train=True, batch_sz=32, source=None):
        self.source = source
        self.datasets = datasets_per_task
        self.loaders = [
                torch.utils.data.DataLoader(x, batch_size=batch_sz, shuffle=True, num_workers=0)
                for x in self.datasets ]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y

        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if type(x) != torch.Tensor:
            # mini_imagenet
            # we assume it's a path --> load from file
            x = self.transform(Image.open(x).convert('RGB'))
            y = torch.Tensor(1).fill_(y).long().squeeze()
        else:
            x = x.float() / 255.
            y = y.long()


        # for some reason mnist does better \in [0,1] than [-1, 1]
        if self.source == 'mnist':
            return x, y
        else:
            return (x - 0.5) * 2, y

def make_valid_from_train(dataset, cpt, fold, n_folds):
    tr_ds, val_ds = [], []
    
    for task_ds in dataset:
        x_t, y_t = task_ds
        size = int(len(x_t) / cpt)
        seg = int(len(x_t)/n_folds)
        spc = int(seg / cpt)
        
        x_tr, y_tr = torch.cat((x_t[:spc*fold], x_t[spc*fold+spc:size])), torch.cat((y_t[:spc*fold], y_t[spc*fold+spc:size]))
        x_val, y_val = x_t[spc*fold:spc*fold+spc], y_t[spc*fold:spc*fold+spc]
        for c in range(1, cpt):
            start_ind = int(c*len(x_t)/cpt)
            x_tr1, y_tr1 = x_t[start_ind:start_ind + spc*fold], y_t[start_ind:start_ind + spc*fold]
            x_v, y_v = x_t[start_ind+spc*fold:start_ind+spc*fold+spc], y_t[start_ind+spc*fold:start_ind+spc*fold+spc]
            x_tr2, y_tr2 = x_t[start_ind+spc*fold+spc:start_ind+ size], y_t[start_ind+spc*fold+spc:start_ind+size]
            
            x_tr, y_tr = torch.cat((x_tr, torch.cat((x_tr1, x_tr2)))), torch.cat((y_tr, torch.cat((y_tr1, y_tr2))))
            x_val, y_val = torch.cat((x_val, x_v)), torch.cat((y_val, y_v))
        
        tr_ds  += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds

def get_split(train, test, fold, n_folds, n_tasks, n_classes, source):
    n_tasks   = n_tasks
    n_classes = n_classes
    assert n_classes % n_tasks == 0
    n_classes_per_task = n_classes / n_tasks
    
    try:
        train_x, train_y = train.data, train.targets
        test_x, test_y = test.data, test.targets
    except:
        train_x, train_y = train.train_data, train.train_labels
        test_x,  test_y  = test.test_data,   test.test_labels
    train_x = np.concatenate((train_x, test_x))
    train_y = np.concatenate((train_y, test_y))
    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            np.stack([elem[i] for elem in out_train]) for i in [0,1] ]


    train_x = torch.Tensor(train_x).permute(0, 3, 1, 2).contiguous()
    train_y = torch.Tensor(train_y)

    # get indices of class split
    train_idx = [((train_y + i) % n_classes).argmax() for i in range(n_classes)]
    train_idx = [0] + [x + 1 for x in sorted(train_idx)]

    train_ds, test_ds = [], []
    skip = n_classes // n_tasks #args.n_tasks
    for i in range(0, n_classes, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        
    train_ds, test_ds = make_valid_from_train(train_ds, n_classes_per_task, fold, n_folds)

    train_ds = map(lambda x : XYDataset(x[0], x[1], **{'source':source}), train_ds)
    test_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':source}), test_ds)

    return train_ds, test_ds

# def get_split_cifar10(train, test):
#     n_tasks   = 5
#     n_classes = 10
#     n_classes_per_task = 2
#     input_size = [3, 32, 32]
    
#     try:
#         train_x, train_y = train.data, train.targets
#         test_x, test_y = test.data, test.targets
#     except:
#         train_x, train_y = train.train_data, train.train_labels
#         test_x,  test_y  = test.test_data,   test.test_labels
#     # sort according to the label
#     out_train = [
#         (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]
    
#     out_test = [
#         (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1])
#     ]

#     train_x, train_y = [
#             np.stack([elem[i] for elem in out_train]) for i in [0,1] ]
    
#     test_x, test_y = [
#             np.stack([elem[i] for elem in out_test]) for i in [0,1] ]


#     train_x = torch.Tensor(train_x).permute(0, 3, 1, 2).contiguous()
#     train_y = torch.Tensor(train_y)
#     test_x = torch.Tensor(test_x).permute(0, 3, 1, 2).contiguous()
#     test_y = torch.Tensor(test_y)
    
#     # get indices of class split
#     train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
#     train_idx = [0] + [x + 1 for x in sorted(train_idx)]
    
#     test_idx = [((test_y + i) % 10).argmax() for i in range(10)]
#     test_idx = [0] + [x + 1 for x in sorted(test_idx)]

#     train_ds, test_ds = [], []
#     skip = 10 // 5 #args.n_tasks
#     for i in range(0, 10, skip):
#         tr_s, tr_e = train_idx[i], train_idx[i + skip]
#         te_s, te_e = test_idx[i], test_idx[i + skip]

#         train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
#         test_ds += [(test_x[te_s:te_e], test_y[te_s:te_e])]

#     train_ds = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), train_ds)
#     test_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), test_ds)

#     return train_ds, test_ds