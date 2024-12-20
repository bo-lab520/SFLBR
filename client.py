import logging
import time
import random

import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset
import torch

from get_model import get_model


class Client(object):

    def __init__(self, conf, eval_dataset, train_dataset, non_iid, id=-1):
        self.client_id = id
        self.conf = conf
        self.poisoner = False
        self.local_model=None

        sub_trainset: Subset = Subset(train_dataset, indices=non_iid)
        self.train_loader = DataLoader(sub_trainset, batch_size=conf["batch_size"], shuffle=True)

        choose_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        choose_index = []
        for i in range(len(eval_dataset.targets)):
            if eval_dataset.targets[i] in choose_labels:
                choose_index.append(i)
        sub_evalset: Subset = Subset(eval_dataset, indices=choose_index)
        self.eval_loader = DataLoader(sub_evalset, batch_size=conf["batch_size"], shuffle=False)

    def set_model(self, model):
        self.local_model = get_model(self.conf["model_name"])
        self.local_model.load_state_dict(model.state_dict())

    def del_model(self):
        del self.local_model

    def local_train(self, global_model):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target, _ = batch
                if self.poisoner:
                    # 无目标攻击，随即置乱标签
                    if self.conf["poison_type"] == "LSA" or 'model replacement' or 'ALIE':
                        shuffle_target = np.array(target)
                        np.random.shuffle(shuffle_target)
                        target = torch.tensor(shuffle_target)
                    # 有目标攻击，将不同类别的特征数据指向同一类别
                    elif self.conf["poison_type"] == "target":
                        # 在MNIST、CIFAR10中，将标签为1的数据指向标签0
                        if self.conf["type"] == "mnist" or self.conf["type"] == "cifar10":
                            for i in range(len(target)):
                                if target[i] == 1:
                                    target[i] = 0
                        # 在CIFAR100中，将标签为1~9的数据指向标签0
                        elif self.conf["type"] == "cifar100":
                            for i in range(len(target)):
                                if 1 <= target[i] <= 9:
                                    target[i] = 0

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()

                _, output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)

                if self.conf["method"] == "myfl":
                    proximal_term = 0.0
                    for w, w_t in zip(self.local_model.parameters(), global_model.parameters()):
                        proximal_term += ((w - w_t).norm(2) ** 2)
                    # print(loss, proximal_term)
                    loss += self.conf['alpha'] * proximal_term

                loss.backward()
                optimizer.step()

            print("Client {} Epoch {} done.".format(self.client_id, e))
        # 计算模型更新

        # print(self.model_eval())

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])

        if self.poisoner and self.conf["poison_type"] == "model replacement":
            for name, data in self.local_model.state_dict().items():
                local_data = self.conf["candidates"]*data - (self.conf["candidates"]-1)*global_model.state_dict()[name]
                diff[name] = local_data - global_model.state_dict()[name]
        elif self.poisoner and self.conf["poison_type"] == "ALIE":
            # server执行
            pass
        elif self.poisoner and self.conf["poison_type"] == "LSA":
            pass

        # 归一化
        if self.conf["method"] == "shieldfl":
            flag = False
            for name, data in diff.items():
                data = data.reshape(-1)
                if flag is not True:
                    vector = data
                    flag = True
                else:
                    vector = torch.cat((vector, data), dim=0)
            vector /= torch.norm(vector)
            start_index = 0
            for name, params in diff.items():
                length = 1
                for l in params.size():
                    length *= l
                _params = vector[start_index:start_index + length]
                start_index += length
                diff[name] = (_params.reshape(params.size()))

        return diff

    def model_eval(self):
        self.local_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target, _ = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.local_model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3, 4])
    if a[0] < 2:
        a[0]=4
    print(a)
