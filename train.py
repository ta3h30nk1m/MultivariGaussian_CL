import numpy as np
import torch
from sklearn.mixture import GaussianMixture as GMM
from dataset import *
from model import *
from utils import *

dtype = torch.cuda.FloatTensor  # run on GPU

class Appr(object):

    def __init__(self, model, nepochs=0, lr=0,  clipgrad=10, args=None):
        self.model = model

        self.nepochs = nepochs
        self.lr = lr
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer(lr = lr)

        self.Pc1 = torch.autograd.Variable(torch.eye(3 * 2 * 2).type(dtype))
        self.Pc2 = torch.autograd.Variable(torch.eye(64 * 2 * 2).type(dtype))
        self.Pc3 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype))
        self.P1 = torch.autograd.Variable(torch.eye(256 * 4 * 4).type(dtype))
        self.P2 = torch.autograd.Variable(torch.eye(1000).type(dtype))
        self.P3 = torch.autograd.Variable(torch.eye(1000).type(dtype))

        self.test_max = 0

        return

    def _get_optimizer(self, t=0, lr=None):
        lr = lr
        lr_owm = lr
        #if lr is None:
        #    lr = self.lr
        #    lr_own = self.lr

        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc3.parameters(), 'lr': lr_owm}
                                     ], lr=lr, momentum=0.9)
        return optimizer

    def train(self, t, tr_loader, val_loader, buffer, Features, total_num):
    #def train(self, t, tr_loader, val_loader, means, covs, Features, total_num):
        best_loss = np.inf
        best_acc = 0
        best_model = get_model(self.model)
        lr = 0.02 if t == 0 else self.lr
        self.optimizer = self._get_optimizer(t, lr)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        nepochs = self.nepochs
        test_max = 0
        # Loop epochs
        try:
            for e in range(nepochs):
                # Train

                self.train_epoch(t, tr_loader, buffer, cur_epoch=e, nepoch=nepochs)
                train_loss, train_acc = self.eval(t, tr_loader, False)
                print('| [{:d}/5], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, e + 1,
                                                                                                 nepochs, train_loss,
                                                                                                 100 * train_acc),
                      end='')
                #self.scheduler.step()
                #print(self.scheduler.get_lr())
                # # Valid
                valid_loss, valid_acc = self.eval(t, val_loader, True)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                if valid_acc>=test_max:
                    test_max = max(test_max, valid_acc)
                    #best_model = get_model(self.model)

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Cur_val_acc : {:2.2f}%<<<'.format(100 * test_max, 100 * valid_acc))

            #set_model_(self.model, best_model)
            best_model = get_model(self.model)
            
            for i, (data, target) in enumerate(tr_loader):
                data, targets = data.cuda(), target.cuda()
                features = self.model.feedforward(data)
                #features = features.cpu().detach().numpy()
                for i in range(len(targets)):
#                     r = torch.reshape(features[i], [features[i].shape[0], 1])
#                     cov = torch.mm(r, r.T)
                    feature = features[i].cpu().detach().numpy()
#                     means[targets[i]] = np.add(means[targets[i]], feature)
#                     covs[targets[i]] = np.add(covs[targets[i]], cov.cpu().detach().numpy())
                    idx = int(total_num[targets[i]])
                    Features[targets[i]][idx] = feature
                    total_num[targets[i]] += 1

            for ind in range((t)*2, (t+1)*2):
                gm = GMM(n_components=3, random_state=42).fit(Features[ind])
                #gm = GaussianMixture(n_components=2, n_features=4096).cuda()
                #gm.fit(Features[ind])
                buffer.append(gm)
            
            if t > 0:
                optimizer = torch.optim.SGD([{'params': self.model.fc1.parameters(), 'lr': 0.01},
                                     {'params': self.model.fc2.parameters(), 'lr': 0.01},
                                     {'params': self.model.fc3.parameters(), 'lr': 0.01}
                                     ], lr=0.01, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
                ne = 60
                for e in range(ne):
                    #set_model_(self.model, best_model)
                    #reg_x, reg_y = regenerate_multivari_feature(t+1, means, covs)
                    reg_x, reg_y = generate(t+1, buffer)
                    reg_x, reg_y = reg_x.cuda(), reg_y.cuda()
                    
                    for i_batch in range(0, len(reg_x), 32):
                        images = reg_x[i_batch:i_batch+32].clone().detach()
                        targets = reg_y[i_batch:i_batch+32].clone().detach()
                        output = self.model.train_classifier(images)
                        loss = self.ce(output, targets)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                    _, train_acc = self.eval_classifier(reg_x, reg_y)
                    print("Accuracy of classifier of train set: ", train_acc*100, "%")
                    
                    valid_loss, valid_acc = self.eval(t, val_loader, True)
                    
                    if valid_acc>=test_max:
                        test_max = max(test_max, valid_acc)
                        best_model = get_model(self.model)
                    #if e % 100 == 0:
                    print('>>>[{:d} Epoch] Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(e+1, 100 * test_max, 100 * valid_acc))
                    #if train_acc==1:
                    #    break
        except KeyboardInterrupt:
            print()

        # Restore best validation model
        set_model_(self.model, best_model)
        return buffer, Features, total_num

    def train_epoch(self, t, tr_loader, means, cur_epoch=0, nepoch=0):
        self.model.train()
        r_len = len(tr_loader)

        # Loop batches
        for i_batch, (data, target) in enumerate(tr_loader):
            data, target = data.cuda(), target.cuda()
            output, h_list, x_list, f = self.model.forward(data)
            loss = self.ce(output, target)
            SSL_loss = 0
            for k in range(4):
                data = torch.rot90(data, 1, dims=[2, 3]).cuda()
                label = torch.LongTensor(np.full(data.shape[0], k)).cuda()
                output = self.model.SSL(data)
                k_loss = self.ce(output, label)
                SSL_loss += k_loss
            SSL_loss /= 4 
            loss += SSL_loss

            self.optimizer.zero_grad()
            loss.backward()
            lamda = i_batch / r_len/nepoch + cur_epoch/nepoch

            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]
            @torch.no_grad()
            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
                if cnn:
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                            #r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p, torch.t(r))
                            p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                            #k = torch.mm(p, torch.t(r)) / (alpha + torch.mm(r, torch.mm(p, torch.t(r))))
                            #p.sub_(torch.mm(k, torch.mm(r, p)))
                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
                else:
                    r = x
                    k = torch.mm(p, torch.t(r))
                    p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    #k = torch.mm(p, torch.t(r)) / (alpha + torch.mm(r, torch.mm(p, torch.t(r))))
                    #p.sub_(torch.mm(k, torch.mm(r, p)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
            # Compensate embedding gradients
            for n, w in self.model.named_parameters():
                if n == 'c1.weight':
                    pro_weight(self.Pc1, x_list[0], w, alpha=alpha_array[0], stride=2)

                if n == 'c2.weight':
                    pro_weight(self.Pc2, x_list[1], w, alpha=alpha_array[0], stride=2)

                if n == 'c3.weight':
                    pro_weight(self.Pc3, x_list[2], w, alpha=alpha_array[0], stride=2)

                if n == 'fc1.weight':
                    pro_weight(self.P1,  h_list[0], w, alpha=alpha_array[1], cnn=False)

                if n == 'fc2.weight':
                    pro_weight(self.P2,  h_list[1], w, alpha=alpha_array[2], cnn=False)

                if n == 'fc3.weight':
                    pro_weight(self.P3,  h_list[2], w, alpha=alpha_array[3], cnn=False)

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        return

    def eval(self, cur_task, loader, validation=True):

        self.model.eval()

        if validation:
            final_loss = 0
            final_acc = 0
            for task, vl_loader in enumerate(loader):
                if task > cur_task: break
                total_loss = 0
                total_acc = 0
                total_num = 0
                for batch, (data, target) in enumerate(vl_loader):
                    data, target = data.cuda(), target.cuda()
                    output,  _, _, _ = self.model.forward(data)
                    loss = self.ce(output, target)
                    _, pred = output.max(1)
                    hits = (pred % 100 == target).float()

                    total_loss += loss.data.cpu().numpy().item() * len(target)
                    total_acc += hits.sum().data.cpu().numpy().item()
                    total_num += len(target)
                total_loss /= total_num
                total_acc /= total_num
                final_loss += total_loss
                final_acc += total_acc
            return final_loss / (cur_task+1), final_acc / (cur_task+1)
        else:
            total_loss = 0
            total_acc = 0
            total_num = 0
            for batch, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()
                output,  _, _, _ = self.model.forward(data)
                loss = self.ce(output, target)
                _, pred = output.max(1)
                hits = (pred % 100 == target).float()

                total_loss += loss.data.cpu().numpy().item() * len(target)
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += len(target)
            return total_loss / total_num, total_acc / total_num
                
    def eval_classifier(self, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), 64):
            b = r[i:min(i + 64, len(r))]
            images = x[b]
            targets = y[b]

            # Forward
            output = self.model.train_classifier(images)
            loss = self.ce(output, targets)
            _, pred = output.max(1)
            hits = (pred % 100 == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num
    
    def calculate_covs(self, means, tr_loader, label):
        covariance = torch.zeros((means.shape[0], means.shape[0])).cuda()
        count = 0
        for _, (data, target) in enumerate(tr_loader):
            data, targets = data.cuda(), target.cuda()
            feature = self.model.feedforward(data)
            #feature = feature.cpu().detach().numpy()
            mean = torch.Tensor(means).cuda()
            for i in range(len(target)):
                if target[i] == label:
                    #r = np.subtract(feature[i], means)
                    r = torch.sub(feature[i], mean)
                    r = torch.reshape(r, [r.shape[0], 1])
                    covariance = torch.add(covariance, torch.mm(r, torch.transpose(r, 0, 1)))
                    count += 1
        covariance /= count
        return covariance.cpu().detach().numpy()
