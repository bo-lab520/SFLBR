import copy
import json
import logging
import random
import time

import numpy as np
import torch

from server import Server
from client import *
import datasets

from scipy.stats import pearsonr
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


def ALIE(server, map_diff, poisoner_nums):
    if len(poisoner_nums)==0:
        return map_diff

    mal_grad=[]
    for i in poisoner_nums:
        diff=map_diff[i]
        vector = None
        for name, data in diff.items():
            data = data.reshape(-1)
            if vector == None:
                vector = data
            else:
                vector = torch.cat((vector, data), dim=0)
        mal_grad.append(np.array(vector.cpu()))

    grad_mean = np.mean(mal_grad, axis=0)
    grad_stdev = np.std(mal_grad, axis=0)
    grad_mean = grad_mean + 1.5 * grad_stdev

    start_index = 0
    for name, params in server.global_model.state_dict().items():
        length = 1
        for l in params.size():
            length *= l
        _params = grad_mean[start_index:start_index + length]
        _params=torch.tensor(_params).cuda()
        start_index += length

        for i in poisoner_nums:
            map_diff[i][name] = (_params.reshape(params.size()))

    return map_diff

def oracle_AGR_detect(honest_map_vec, avg_honest_vec, uint_poison_vec, lamda):
    poison_vec=avg_honest_vec+lamda*uint_poison_vec

    max_poison_dis=0.
    max_honest_dis=0.
    keys = honest_map_vec.keys()
    for i in range(len(keys)):
        dis=np.linalg.norm(poison_vec-honest_map_vec[keys[i]], ord=2)
        if dis>max_poison_dis:
            max_poison_dis=dis

        for j in range(i+1, len(keys)):
            dis=np.linalg.norm(honest_map_vec[keys[i]-honest_map_vec[keys[j]]], ord=2)
            if dis>max_honest_dis:
                max_honest_dis=dis

    if max_poison_dis<=max_honest_dis:
        return True
    return False

def AGR(server, map_diff, poisoner_nums):
    honest_map_vec={}
    avg_honest_vec=None
    honest_nums=len(map_diff.keys())-len(poisoner_nums)

    for i in map_diff.keys():
        if i not in poisoner_nums:

            diff = map_diff[i]
            vector = None
            for name, data in diff.items():
                data = data.reshape(-1)
                if vector == None:
                    vector = data
                else:
                    vector = torch.cat((vector, data), dim=0)
            honest_map_vec[i]=np.array(vector.cpu())
            avg_honest_vec=np.zeros_like(honest_map_vec[i])

    for i in honest_map_vec.keys():
        avg_honest_vec+=honest_map_vec[i]
    avg_honest_vec=avg_honest_vec/honest_nums
    unit_honest_vec=avg_honest_vec/np.linalg.norm(avg_honest_vec, ord=2)
    unit_poison_vec=-unit_honest_vec

    lamda=20.
    step=lamda/2
    lamda_succ=0
    threshld=1e-5

    while abs(lamda_succ-lamda)>threshld:
        if oracle_AGR_detect(honest_map_vec, avg_honest_vec, unit_poison_vec, lamda):
            lamda_succ=lamda
            lamda=lamda+step/2
        else:
            lamda=lamda-step/2
        step=step/2

    poison_vec=avg_honest_vec+lamda*unit_poison_vec

    start_index = 0
    for name, params in server.global_model.state_dict().items():
        length = 1
        for l in params.size():
            length *= l
        _params = poison_vec[start_index:start_index + length]
        start_index += length

        for i in poisoner_nums:
            map_diff[i][name] = _params.reshape(params.size())

    return map_diff


def main(conf):

    filename = conf["method"] + ',' + conf['model_name'] + ',' + conf["type"] + ",n=" + str(conf["clients"]) + '!' + str(conf["candidates"]) + \
               ",m=" + str(conf["poisoner_rate"]) + ',' + conf["poison_type"] + ",d_alpha=" + str(
        conf["dirichlet_alpha"])

    if conf["method"]=='myfl':
        filename+=',alpha='+str(conf['alpha'])
    filename+='.log'

    logging.basicConfig(level=logging.INFO,
                        filename="log/CIFAR100/ALIE,0point05/" + filename,
                        filemode='w')

    data_root="F:/code/data/"+conf["type"]
    train_datasets, eval_datasets = datasets.get_dataset(data_root, conf["type"])

    server = Server(conf, eval_datasets)
    clients = []
    # non-IID数据
    client_idx = datasets.dirichlet_nonIID_data(train_datasets, conf)

    # 选其中前20个类别进行试验，筛选索引
    # choose_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # for key in client_idx.keys():
    #     dataset_index = client_idx[key]
    #     choose_index = []
    #     for i in dataset_index:
    #         if train_datasets.targets[i] in choose_labels:
    #             choose_index.append(i)
    #     client_idx[key]=choose_index

    for c in range(conf["clients"]):
        clients.append(Client(conf, eval_datasets, train_datasets, client_idx[c + 1], c + 1))

    random.seed(1234)
    poisoner = random.sample(clients, int(conf["clients"] * conf["poisoner_rate"]))
    # print(poisoner)
    for p in poisoner:
        p.poisoner = True

    all_acc = []
    all_f_acc = []
    all_o_acc = []
    max_acc = 0
    max_f_acc = 0
    max_o_acc = 0

    m_trust_scores = []
    h_trust_scores = []

    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)

        if e == 40:
            conf["lr"] *= 0.1
        if e == 80:
            conf["lr"] *= 0.1

        candidates = random.sample(clients, conf["candidates"])

        # sent model
        for c in candidates:
            c.set_model(server.global_model)

        poisoner_nums=[]
        for c in candidates:
            if c.poisoner:
                poisoner_nums.append(c.client_id)

        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params).cuda()

        map_diff = {}
        if conf["method"] == "fedavg":
            for c in candidates:
                diff = c.local_train(server.global_model)
                map_diff[c.client_id]=diff
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff=AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                for name, data in diff.items():
                    if data.type() != weight_accumulator[name].type():
                        weight_accumulator[name].add_(data.to(torch.int64))
                    else:
                        weight_accumulator[name].add_(data)

            server.model_aggregate(weight_accumulator, conf["candidates"])
        elif conf["method"] == "krum":
            grad = {}
            grad2vector = {}

            for c in candidates:
                diff = c.local_train(server.global_model)
                map_diff[c.client_id]=diff
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff=AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                grad[key] = diff
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                grad2vector[key] = vector
            closest_nums = conf["candidates"] - len(poisoner_nums) - 2
            euclidean_distance = {}
            for id, vec in grad2vector.items():
                distances = []
                for _id, _vec in grad2vector.items():
                    if id != _id:
                        distances.append(torch.norm(vec - _vec))
                distances.sort()
                sum_dis = 0.
                for i in range(closest_nums):
                    sum_dis += distances[i]
                euclidean_distance[id] = sum_dis
            target_id = -1
            min_dis = pow(2, 31)
            for id, dis in euclidean_distance.items():
                if dis < min_dis:
                    min_dis = dis
                    target_id = id
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(grad[target_id][name])
            server.model_aggregate(weight_accumulator, 1)
        elif conf["method"] == "trimmed_mean":
            if conf["candidates"] - 2 * len(poisoner_nums) <= 0:
                continue
            grad2vector = []

            for c in candidates:
                diff = c.local_train(server.global_model)
                map_diff[c.client_id]=diff
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff=AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                vector=np.array(vector.cpu())
                grad2vector.append(vector)
            sorted_vector = np.sort(grad2vector, axis=0)

            sorted_vector = sorted_vector[len(poisoner_nums):conf["candidates"] - len(poisoner_nums)]
            trim_mean_vector=np.sum(sorted_vector, axis=0)/(conf["candidates"] - 2 * len(poisoner_nums))

            start_index = 0
            for name, params in server.global_model.state_dict().items():
                length = 1
                for l in params.size():
                    length *= l
                _params = trim_mean_vector[start_index:start_index + length]
                _params = torch.tensor(_params).cuda()
                if _params.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_((_params.reshape(params.size())).to(torch.int64))
                else:
                    weight_accumulator[name].add_(_params.reshape(params.size()))
                start_index += length
            server.model_aggregate(weight_accumulator, 1)
        elif conf["method"] == "median":
            grad2vector = []

            for c in candidates:
                diff = c.local_train(server.global_model)

                map_diff[c.client_id]=diff
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff=AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                vector=np.array(vector.cpu())
                grad2vector.append(vector)
            # median_vector = torch.zeros_like(vector)
            sorted_vector=np.sort(grad2vector, axis=0)
            length=len(sorted_vector)
            if length & 1:
                median_vector=sorted_vector[int(length/2)]
            else:
                median_vector=(sorted_vector[int(length/2)-1]+sorted_vector[int(length/2)])/2

            start_index = 0
            for name, params in server.global_model.state_dict().items():
                length = 1
                for l in params.size():
                    length *= l
                _params = median_vector[start_index:start_index + length]
                _params=torch.tensor(_params).cuda()
                start_index += length
                if _params.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_((_params.reshape(params.size())).to(torch.int64))
                else:
                    weight_accumulator[name].add_(_params.reshape(params.size()))
            server.model_aggregate(weight_accumulator, 1)
        elif conf["method"] == "pefl":
            grad = {}
            grad_vector = {}
            grad2vector = []

            for c in candidates:
                diff = c.local_train(server.global_model)
                map_diff[c.client_id]=diff
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff=AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                vector=np.array(vector.cpu())
                grad2vector.append(vector)
                grad[key] = diff
                grad_vector[key] = vector
            sorted_vector = np.sort(grad2vector, axis=0)
            length = len(sorted_vector)
            if length & 1:
                median_vector = sorted_vector[int(length / 2)]
            else:
                median_vector = (sorted_vector[int(length / 2) - 1] + sorted_vector[int(length / 2)]) / 2

            pearson_similarity = {}
            for _id, vec in grad_vector.items():
                ps = pearsonr(vec, median_vector)
                pearson_similarity[_id] = ps[0]
            mu = {}
            all_mu = 0
            for _id, ps in pearson_similarity.items():
                if 1 - ps==0:
                    mu[_id] = 1000
                else:
                    mu[_id] = max(0, np.log((1 + ps) / (1 - ps)) - 0.5)
                all_mu += mu[_id]
            if all_mu == 0:
                continue
            for _id, diff in grad.items():
                for name, data in diff.items():
                    if data.type() != weight_accumulator[name].type():
                        weight_accumulator[name].add_(((mu[_id] / all_mu) * data).to(torch.int64))
                    else:
                        if data.dtype==torch.int64:
                            weight_accumulator[name].add_(((mu[_id] / all_mu) * data).to(torch.int64))
                        else:
                            weight_accumulator[name].add_((mu[_id] / all_mu) * data)

            server.model_aggregate(weight_accumulator, 1)
        elif conf["method"] == "shieldfl":
            cos = torch.nn.CosineSimilarity(dim=-1)
            grad = {}
            grad_vector = {}

            for c in candidates:
                diff = c.local_train(server.global_model)
                map_diff[c.client_id]=diff
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff=AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                if (torch.norm(vector) - 1.0) < 1e-5:
                    grad[key] = diff
                    grad_vector[key] = vector
            if e == 0:
                # global_grad = torch.zeros_like(vector)

                for _id, diff in grad.items():
                    for name, data in diff.items():
                        if data.type() != weight_accumulator[name].type():
                            weight_accumulator[name].add_(data.to(torch.int64))
                        else:
                            weight_accumulator[name].add_(data)
                server.model_aggregate(weight_accumulator, conf['candidates'])
                global_grad = torch.zeros_like(vector)
                for _id, vec in grad_vector.items():
                    global_grad += ((1/conf['candidates']) * vec)

            else:
                min_cos_similarity = 2
                baseline_grad = torch.zeros_like(vector)
                for _id, vec in grad_vector.items():
                    cur_cos = cos(vec, global_grad)
                    if cur_cos < min_cos_similarity:
                        min_cos_similarity = cur_cos
                        baseline_grad = vec
                mu = {}
                all_mu = 0.
                for _id, vec in grad_vector.items():
                    mu[_id] = 1 - cos(baseline_grad, vec)
                    all_mu += mu[_id]
                if all_mu == 0.:
                    continue
                for _id, diff in grad.items():
                    for name, data in diff.items():
                        if data.type() != weight_accumulator[name].type():
                            weight_accumulator[name].add_(((mu[_id] / all_mu) * data).to(torch.int64))
                        else:
                            if data.dtype == torch.int64:
                                weight_accumulator[name].add_(((mu[_id] / all_mu) * data).to(torch.int64))
                            else:
                                weight_accumulator[name].add_((mu[_id] / all_mu) * data)
                server.model_aggregate(weight_accumulator, 1)
                global_grad = torch.zeros_like(vector)
                for _id, vec in grad_vector.items():
                    global_grad += ((mu[_id] / all_mu) * vec)
        elif conf["method"] == "myfl":
            cos = torch.nn.CosineSimilarity(dim=-1)
            grad = {}
            grad_vector = {}
            grad2vector = []

            for c in candidates:
                diff = c.local_train(server.global_model)
                map_diff[c.client_id]=diff
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff=AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                vector=np.array(vector.cpu())
                grad2vector.append(vector)
                grad[key] = diff
                grad_vector[key] = vector
            sorted_vector = np.sort(grad2vector, axis=0)
            length = len(sorted_vector)
            if length & 1:
                median_vector = sorted_vector[int(length / 2)]
            else:
                median_vector = (sorted_vector[int(length / 2) - 1] + sorted_vector[int(length / 2)]) / 2

            start_index = 0
            median_grad = {}
            for name, params in server.global_model.state_dict().items():
                length = 1
                for l in params.size():
                    length *= l
                _params = median_vector[start_index:start_index + length]
                median_grad[name] = torch.tensor(_params).cuda()
                start_index += length
            # 按层计算相似度
            mu = {}
            all_mu = 0.
            m_mus=[]
            h_mus=[]

            for _id, diff in grad.items():
                sum_layer_cos_similarity = 0
                for name, data in diff.items():
                    vector = data.reshape(-1)
                    median_vector = median_grad[name]
                    layer_cos_similarity = cos(vector, median_vector)
                    if layer_cos_similarity >= 0:
                        layer_cos_similarity += 1
                    else:
                        layer_cos_similarity = (layer_cos_similarity + 1) ** 2
                    sum_layer_cos_similarity += layer_cos_similarity
                mu[_id] = sum_layer_cos_similarity.item()
                all_mu += mu[_id]
                # 记录trust scores
                if _id in poisoner_nums:
                    m_mus.append(mu[_id])
                else:
                    h_mus.append(mu[_id])

            if len(m_mus)==0:
                m_trust_scores.append(0.)
            else:
                m_trust_scores.append(np.mean(m_mus)/all_mu)
            if len(h_mus)==0:
                h_trust_scores.append(0.)
            else:
                h_trust_scores.append(np.mean(h_mus)/all_mu)

            for _id, diff in grad.items():
                for name, data in diff.items():
                    if data.type() != weight_accumulator[name].type():
                        weight_accumulator[name].add_(((mu[_id] / all_mu) * data).to(torch.int64))
                    else:
                        if data.dtype == torch.int64:
                            weight_accumulator[name].add_(((mu[_id] / all_mu) * data).to(torch.int64))
                        else:
                            weight_accumulator[name].add_((mu[_id] / all_mu) * data)
            server.model_aggregate(weight_accumulator, 1)

        else:
            print("method is unexisting!")

        acc = server.model_eval()

        for c in candidates:
            c.del_model()

        if conf["poison_type"] == "LSA" or 'model replacement' or 'ALIE':
            if acc > max_acc:
                max_acc = acc
            all_acc.append(acc)
            print("Global Epoch %d, acc: %f\n" % (e, acc))
            logging.info("Global Epoch %d, acc: %f\n" % (e, acc))
        else:
            all_f_acc.append(acc[0])
            if acc[0] > max_f_acc:
                max_f_acc = acc[0]
            all_o_acc.append(acc[1])
            if acc[1] > max_o_acc:
                max_o_acc = acc[1]
            print("Global Epoch %d, f_acc: %f, o_acc: %f\n" % (e, acc[0], acc[1]))
            logging.info("Global Epoch %d, f_acc: %f, o_acc: %f\n" % (e, acc[0], acc[1]))

    if conf["poison_type"] == 'LSA' or 'model replacement' or 'ALIE':
        print(all_acc)
        logging.info(str(all_acc))
        print(max_acc)
        logging.info('max:{}'.format(max_acc))
        print(np.mean(np.array(all_acc)))
        logging.info('Average:{}'.format(np.mean(np.array(all_acc))))
    else:
        print(all_f_acc)
        print(all_o_acc)
        logging.info(str(all_f_acc))
        logging.info(str(all_o_acc))
        print(max_f_acc)
        logging.info(str(max_f_acc))
        print(max_o_acc)
        logging.info(str(max_o_acc))

    if conf["method"] == "myfl":
        print(m_trust_scores)
        logging.info('malcious average trust scores:{}'.format(str(m_trust_scores)))
        print(h_trust_scores)
        logging.info('honest average trust scores:{}'.format(str(h_trust_scores)))


if __name__ == '__main__':

    with open("conf.json", 'r') as f:
        conf = json.load(f)

    main(conf)

    # rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    # rates = [0.4, 0.5]
    # for r in rates:
    #     conf["poisoner_rate"] = r
    #     main(conf)
