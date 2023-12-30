import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler


##############################################################################
# General server function
##############################################################################

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        if 'fedlaw' in args.server_method:
            client_params.append(client_nodes[idx].model.get_param(clone = True))
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    
    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params

def get_model_updates(client_params, prev_para):
    prev_param = copy.deepcopy(prev_para)
    client_updates = []
    for param in client_params:
        client_updates.append(param.sub(prev_param))
    return client_updates

def get_client_params_with_serverlr(server_lr, prev_param, client_updates):
    client_params = []
    with torch.no_grad():
        for update in client_updates:
            param = prev_param.add(update*server_lr)
            client_params.append(param)
    return client_params

##############################################################################
# Proposed function (FedAvg, FedDF, FedBE, FedDyn, FedAdam, Finetune, etc.)
##############################################################################

def proposed_generate_global_model(args, gamma ,agg_weights, client_params, central_node):
    # update the global model with layer-wise aggregation weights
    if "layer" in args.server_method:
        global_params = copy.deepcopy(client_params[0])
        j = 0
        for name_param in global_params:
            param = torch.zeros_like(global_params[name_param])
            for i in range(len(client_params)):
                param += gamma * client_params[i][name_param] * agg_weights[i][j]
            global_params[name_param] = param
            j += 1
        central_node.model.load_state_dict(global_params)
    elif "model" in args.server_method:
        global_params = copy.deepcopy(client_params[0])
        for name_param in global_params:
            param = torch.zeros_like(global_params[name_param])
            for i in range(len(client_params)):
                param += gamma * client_params[i][name_param] * agg_weights[i]
            global_params[name_param] = param
        central_node.model.load_state_dict(global_params)
    elif args.server_method == 'fedavg' or args.server_method == 'uniform':
        global_params = copy.deepcopy(client_params[0])
        for name_param in global_params:
            param = torch.zeros_like(global_params[name_param])
            for i in range(len(client_params)):
                param += gamma * client_params[i][name_param] * agg_weights[i]
            global_params[name_param] = param
        central_node.model.load_state_dict(global_params)
    else:
        raise ValueError('Undefined server method...')
    return central_node

def proposed_optimization(args, agg_weights, client_params, central_node, data, select_list):
    # calculate aggregation weights
    if args.server_method == 'gl_layer_gradients_distance':
        print("global-local layer-wise gradients distance 구현")
        agg_weights = gl_layer_gradients_distance(central_node, client_params)
    elif args.server_method == 'gl_model_gradients_distance':
        print("global-local model-wise gradients distance 구현")
        agg_weights = gl_model_gradients_distance(central_node, client_params)
    elif args.server_method == 'll_layer_gradients_distance':
        print("local-local layer-wise gradients distance 구현")
        # agg_weights = ll_layer_gradients_distance(central_node, client_params)
        agg_weights = ll_layer_gradients_distance(central_node, client_params, agg_weights, select_list)
    elif args.server_method == 'll_model_gradients_distance':
        print("local-local model-wise gradients distance 구현")
        # agg_weights = ll_model_gradients_distance(central_node, client_params)
        agg_weights = ll_model_gradients_distance(central_node, client_params, agg_weights, select_list)
    elif args.server_method == 'gl_layer_cosine':
        print("global-local layer-wise cosine 구현")
        agg_weights = gl_layer_cosine(central_node, client_params)
    elif args.server_method == 'gl_model_cosine':
        print("global-local model-wise cosine 구현")
        agg_weights = gl_model_cosine(central_node, client_params)
    elif args.server_method == 'll_layer_cosine':
        print("local-local layer-wise cosine 구현")
        agg_weights = ll_layer_cosine(central_node, client_params)
    elif args.server_method == 'll_model_cosine':
        print("local-local model-wise cosine 구현")
        agg_weights = ll_model_cosine(central_node, client_params)
    elif args.server_method == 'gl_model_output_proxy':
        print("global-local output distance with proxy data 구현")
        agg_weights = gl_model_output_proxy(central_node, client_params)
    elif args.server_method == 'gl_model_output_snd':
        print("global-local output distance with standard normal distribution noise 구현")
        agg_weights = gl_model_output_snd(central_node, client_params)
    elif args.server_method == 'll_model_output_proxy':
        print("local-local output distance with proxy data 구현")
        agg_weights = ll_model_output_proxy(central_node, client_params)
    elif args.server_method == 'll_model_output_snd':
        print("local-local output distance with standard normal distribution noise 구현")
        agg_weights = ll_model_output_snd(central_node, client_params)
    elif args.server_method == 'model_acc_proxy':
        print("accuracy based aggregation with proxy data 구현")
        agg_weights = model_acc_proxy(central_node, client_params)
    elif args.server_method == 'model_class':
        print("client's class 구현")
        agg_weights = model_class(central_node, client_params, data, select_list)
    elif args.server_method == 'fedavg':
        print("기본 FedAvg")
        agg_weights = agg_weights
    elif args.server_method == 'uniform':
        print("uniform averaging")
        for i in range(len(client_params)):
            agg_weights[i] = 1 / len(client_params)
    else:
        raise ValueError('Undefined server method...')
    if args.fixed_gamma:
        gamma = float(args.fixed_gamma)
    else:
        gamma = 1.0
    return gamma, agg_weights

def gl_layer_gradients_distance(central_node, client_params):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    gradients = []
    for i in range(len(client_params)):
        param = copy.deepcopy(global_model)
        for name_param in param:
            param[name_param] = global_model[name_param] - client_params[i][name_param]
        gradients.append(copy.deepcopy(param))

    # how many layer?
    layer = 0
    for i in global_model:
        layer += 1

    # layer-wise distance calculate
    layer_distance = []
    for i in range(len(gradients)):
        tmp = []
        for name_param in gradients[0]:
            tmp.append(torch.linalg.vector_norm(gradients[i][name_param], ord=2).item())
        layer_distance.append(tmp)

    # aggregation weights calculate
    agg_weights = [[0] * layer for _ in range(len(gradients))]
    for i in range(layer):
        layer_sum = 0
        for j in range(len(gradients)):
            layer_sum += (1 / layer_distance[j][i])
        for j in range(len(gradients)):
            agg_weights[j][i] = (1 / layer_distance[j][i]) / layer_sum

    return agg_weights

def gl_model_gradients_distance(central_node, client_params):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    gradients = []
    for i in range(len(client_params)):
        param = copy.deepcopy(global_model)
        for name_param in param:
            param[name_param] = global_model[name_param] - client_params[i][name_param]
        gradients.append(copy.deepcopy(param))

    # gradients concatenate
    flatted_gradients = []
    for i in range(len(gradients)):
        tmp = None
        for name_param in gradients[i]:
            if tmp == None:
                tmp = torch.flatten(copy.deepcopy(gradients[i][name_param]))
            else:
                tmp = torch.cat((tmp, torch.flatten(copy.deepcopy(gradients[i][name_param]))))
        flatted_gradients.append(tmp)

    # model-wise distance calculate
    model_distance = []
    for i in range(len(flatted_gradients)):
        model_distance.append(torch.linalg.vector_norm(flatted_gradients[i], ord=2).item())

    # aggregation weights calculate
    model_distance = torch.tensor(model_distance)
    agg_weights = (1 / model_distance) / sum(1 / model_distance)
    return agg_weights

# local-local gradients들 하나의 gradeitns vector 로 합치고 이와의 차이를 이용하는 방법
def ll_layer_gradients_distance(central_node, client_params, fedavg_weights, select_list):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    gradients = []
    for i in range(len(client_params)):
        param = copy.deepcopy(global_model)
        for name_param in param:
            param[name_param] = global_model[name_param] - client_params[i][name_param]
        gradients.append(copy.deepcopy(param))

    # averaged local gradients calculate
    for i in range(len(gradients)):
        if i == 0:
            param = copy.deepcopy(gradients[i])
            for name_param in param:
                param[name_param] = param[name_param] * fedavg_weights[i]
                # param[name_param] = param[name_param]
        else:
            for name_param in param:
                param[name_param] = param[name_param] + (gradients[i][name_param] * fedavg_weights[i])
                # param[name_param] = param[name_param] + gradients[i][name_param]
    # for name_param in param:
    #     param[name_param] = param[name_param] / len(gradients)
    avg_gradients = param

    # how many layer?
    layer = 0
    for i in global_model:
        layer += 1

    # local-local layer-wise distance calculate
    layer_distance = []
    for i in range(len(gradients)):
        tmp_i = []
        for name_param in gradients[0]:
            tmp_i.append(torch.linalg.vector_norm(avg_gradients[name_param] - gradients[i][name_param], ord=2).item())
        layer_distance.append(tmp_i)

    # aggregation weights calculate
    agg_weights = [[0] * layer for _ in range(len(gradients))]
    for i in range(layer):
        layer_sum = 0
        for j in range(len(gradients)):
            layer_sum += (1 / layer_distance[j][i])
        for j in range(len(gradients)):
            agg_weights[j][i] = (1 / layer_distance[j][i]) / layer_sum

    return agg_weights

def ll_model_gradients_distance(central_node, client_params, fedavg_weights, select_list):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    gradients = []
    for i in range(len(client_params)):
        param = copy.deepcopy(global_model)
        for name_param in param:
            param[name_param] = global_model[name_param] - client_params[i][name_param]
        gradients.append(copy.deepcopy(param))

    # averaged local gradients calculate
    for i in range(len(gradients)):
        if i == 0:
            param = copy.deepcopy(gradients[i])
            for name_param in param:
                param[name_param] = param[name_param] * fedavg_weights[i]
                # param[name_param] = param[name_param]
        else:
            for name_param in param:
                param[name_param] = param[name_param] + (gradients[i][name_param] * fedavg_weights[i])
                # param[name_param] = param[name_param] + gradients[i][name_param]
    # for name_param in param:
    #     param[name_param] = param[name_param] / len(gradients)
    avg_gradients = param

    # gradients concatenate
    flatted_avg_gradients = []
    for name_param in avg_gradients:
        if len(flatted_avg_gradients) == 0:
            flatted_avg_gradients = torch.flatten(avg_gradients[name_param])
        else:
            flatted_avg_gradients = torch.cat((flatted_avg_gradients, torch.flatten(avg_gradients[name_param])))
    flatted_gradients = []
    for i in range(len(gradients)):
        tmp = None
        for name_param in gradients[i]:
            if tmp == None:
                tmp = torch.flatten(gradients[i][name_param])
            else:
                tmp = torch.cat((tmp, torch.flatten(copy.deepcopy(gradients[i][name_param]))))
        flatted_gradients.append(tmp)

    # local-local model-wise distance calculate
    model_distance = []
    for i in range(len(gradients)):
        model_distance.append(torch.linalg.vector_norm(flatted_avg_gradients - flatted_gradients[i], ord=2).item())

    # aggregation weights calculate
    mean_model_distance = torch.tensor(model_distance)
    agg_weights = (1 / mean_model_distance) / sum(1 / mean_model_distance)

    return agg_weights

# # local-local gradients distance 평균 내는 방법
# def ll_layer_gradients_distance(central_node, client_params):
#     global_model = copy.deepcopy(central_node.model.state_dict())
#
#     # local gradients calculate
#     gradients = []
#     for i in range(len(client_params)):
#         param = copy.deepcopy(global_model)
#         for name_param in param:
#             param[name_param] = global_model[name_param] - client_params[i][name_param]
#         gradients.append(copy.deepcopy(param))
#
#     # how many layer?
#     layer = 0
#     for i in global_model:
#         layer += 1
#
#     # local-local layer-wise distance calculate
#     layer_distance = []
#     for i in range(len(gradients)):
#         tmp_i = []
#         for j in range(len(gradients)):
#             tmp_j = []
#             for name_param in gradients[0]:
#                 tmp_j.append(torch.linalg.vector_norm(gradients[i][name_param] - gradients[j][name_param], ord=2).item())
#             tmp_i.append(tmp_j)
#         layer_distance.append(tmp_i)
#
#     # layer-wise distance average calculate
#     mean_layer_distance = []
#     for i in range(len(gradients)):
#         mean = []
#         for k in range(layer):
#             layer_mean = 0.0
#             for j in range(len(gradients)):
#                 layer_mean += layer_distance[i][j][k]
#             mean.append(layer_mean / len(gradients))
#         mean_layer_distance.append(mean)
#
#     # aggregation weights calculate
#     agg_weights = [[0] * layer for _ in range(len(gradients))]
#     for i in range(layer):
#         layer_sum = 0
#         for j in range(len(gradients)):
#             layer_sum += (1 / mean_layer_distance[j][i])
#         for j in range(len(gradients)):
#             agg_weights[j][i] = (1 / mean_layer_distance[j][i]) / layer_sum
#
#     return agg_weights
#
# def ll_model_gradients_distance(central_node, client_params):
#     global_model = copy.deepcopy(central_node.model.state_dict())
#
#     # local gradients calculate
#     gradients = []
#     for i in range(len(client_params)):
#         param = copy.deepcopy(global_model)
#         for name_param in param:
#             param[name_param] = global_model[name_param] - client_params[i][name_param]
#         gradients.append(copy.deepcopy(param))
#
#     # gradients concatenate
#     flatted_gradients = []
#     for i in range(len(gradients)):
#         tmp = None
#         for name_param in gradients[i]:
#             if tmp == None:
#                 tmp = torch.flatten(copy.deepcopy(gradients[i][name_param]))
#             else:
#                 tmp = torch.cat((tmp, torch.flatten(copy.deepcopy(gradients[i][name_param]))))
#         flatted_gradients.append(tmp)
#
#     # local-local model-wise distance calculate
#     model_distance = []
#     for i in range(len(gradients)):
#         tmp = []
#         for j in range(len(gradients)):
#             tmp.append(torch.linalg.vector_norm(flatted_gradients[i] - flatted_gradients[j], ord=2).item())
#         model_distance.append(tmp)
#
#     # model-wise distance average calculate
#     mean_model_distance = []
#     for i in range(len(gradients)):
#         mean = 0.0
#         for j in range(len(gradients)):
#             mean += model_distance[i][j]
#         mean_model_distance.append(mean / len(gradients))
#
#     # aggregation weights calculate
#     mean_model_distance = torch.tensor(mean_model_distance)
#     agg_weights = (1 / mean_model_distance) / sum(1 / mean_model_distance)
#
#     return agg_weights

def gl_layer_cosine(central_node, client_params):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # how many layer?
    layer = 0
    for i in global_model:
        layer += 1

    # global-local layer-wise weights cosine similarity calculate
    cosine_similarity = []
    for i in range(len(client_params)):
        tmp = []
        for name_param in global_model:
            inner = (global_model[name_param] * client_params[i][name_param]).sum()
            norm1 = torch.linalg.vector_norm(global_model[name_param], ord=2)
            norm2 = torch.linalg.vector_norm(client_params[i][name_param], ord=2)
            # 추후 문제 있을 수 있음, 문제가 있다면 먼저 살펴보기
            if norm1 == 0 or norm2 == 0:
                tmp.append(0.0)
            else:
                tmp.append((inner / (norm1 * norm2)).item())
        cosine_similarity.append(tmp)

    # normalize cosine similarity -1~1 -> 0~1 for convert it to aggregation weights
    for i in range(len(cosine_similarity)):
        for j in range(len(cosine_similarity[i])):
            cosine_similarity[i][j] = (cosine_similarity[i][j] + 1) / 2

    # aggregation weights calculate
    agg_weights = [[0] * layer for _ in range(len(client_params))]
    for i in range(layer):
        layer_sum = 0
        for j in range(len(client_params)):
            layer_sum += cosine_similarity[j][i]
        for j in range(len(client_params)):
            agg_weights[j][i] = cosine_similarity[j][i] / layer_sum
    agg_weights = torch.tensor(agg_weights)
    return agg_weights

def gl_model_cosine(central_node, client_params):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # weights concatenate
    flatted_weights = []
    for i in range(len(client_params)):
        tmp = None
        for name_param in client_params[i]:
            if tmp == None:
                tmp = torch.flatten(copy.deepcopy(client_params[i][name_param]))
            else:
                tmp = torch.cat((tmp, torch.flatten(copy.deepcopy(client_params[i][name_param]))))
        flatted_weights.append(tmp)
    flatted_global = None
    for name_param in global_model:
        if flatted_global == None:
            flatted_global = torch.flatten(copy.deepcopy(global_model[name_param]))
        else:
            flatted_global = torch.cat((flatted_global, torch.flatten(copy.deepcopy(global_model[name_param]))))

    # global-local model-wise weights cosine similarity calculate
    cosine_similarity = []
    for i in range(len(client_params)):
        inner = (flatted_global * flatted_weights[i]).sum()
        norm1 = torch.linalg.vector_norm(flatted_global, ord=2)
        norm2 = torch.linalg.vector_norm(flatted_weights[i], ord=2)
        if norm1 == 0 or norm2 == 0:
            cosine_similarity.append(0.0)
        else:
            cosine_similarity.append((inner / (norm1 * norm2)).item())

    # normalize cosine similarity -1~1 -> 0~1 for convert it to aggregation weights
    for i in range(len(cosine_similarity)):
        cosine_similarity[i] = (cosine_similarity[i] + 1) / 2

    # aggregation weights calculate
    cosine_similarity = torch.tensor(cosine_similarity)
    agg_weights = cosine_similarity / sum(cosine_similarity)

    return agg_weights

def ll_layer_cosine(central_node, client_params):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    gradients = []
    for i in range(len(client_params)):
        param = copy.deepcopy(global_model)
        for name_param in param:
            param[name_param] = global_model[name_param] - client_params[i][name_param]
        gradients.append(copy.deepcopy(param))

    # how many layer?
    layer = 0
    for i in global_model:
        layer += 1

    # local-local layer-wise gradients cosine similarity calculate & normalize
    cosine_similarity = []
    for i in range(len(client_params)):
        tmp_i = []
        for j in range(len(client_params)):
            tmp_j = []
            for name_param in client_params[0]:
                inner = (client_params[i][name_param] * client_params[j][name_param]).sum()
                norm1 = torch.linalg.vector_norm(client_params[i][name_param], ord=2)
                norm2 = torch.linalg.vector_norm(client_params[j][name_param], ord=2)
                # 추후 문제 있을 수 있음, 문제가 있다면 먼저 살펴보기
                if norm1 == 0 or norm2 == 0:
                    tmp_j.append((0.0 + 1) / 2)
                else:
                    tmp_j.append((((inner / (norm1 * norm2)).item()) + 1) / 2)
            tmp_i.append(tmp_j)
        cosine_similarity.append(tmp_i)

    # layer-wise cosine average calculate
    mean_layer_cosine = []
    for i in range(len(cosine_similarity)):
        mean = []
        for k in range(layer):
            layer_mean = 0.0
            for j in range(len(cosine_similarity)):
                layer_mean += cosine_similarity[i][j][k]
            mean.append(layer_mean / len(cosine_similarity))
        mean_layer_cosine.append(mean)

    # aggregation weights calculate
    agg_weights = [[0] * layer for _ in range(len(gradients))]
    for i in range(layer):
        layer_sum = 0
        for j in range(len(gradients)):
            layer_sum += mean_layer_cosine[j][i]
        for j in range(len(gradients)):
            agg_weights[j][i] = mean_layer_cosine[j][i] / layer_sum

    return agg_weights

def ll_model_cosine(central_node, client_params):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    gradients = []
    for i in range(len(client_params)):
        param = copy.deepcopy(global_model)
        for name_param in param:
            param[name_param] = global_model[name_param] - client_params[i][name_param]
        gradients.append(copy.deepcopy(param))

    # gradients concatenate
    flatted_gradients = []
    for i in range(len(gradients)):
        tmp = None
        for name_param in gradients[i]:
            if tmp == None:
                tmp = torch.flatten(copy.deepcopy(gradients[i][name_param]))
            else:
                tmp = torch.cat((tmp, torch.flatten(copy.deepcopy(gradients[i][name_param]))))
        flatted_gradients.append(tmp)

    # local-local model-wise gradients cosine similarity calculate & normalize
    cosine_similarity = []
    for i in range(len(client_params)):
        tmp_i = []
        for j in range(len(client_params)):
            inner = (flatted_gradients[i] * flatted_gradients[j]).sum()
            norm1 = torch.linalg.vector_norm(flatted_gradients[i], ord=2)
            norm2 = torch.linalg.vector_norm(flatted_gradients[j], ord=2)
            # 추후 문제 있을 수 있음, 문제가 있다면 먼저 살펴보기
            if norm1 == 0 or norm2 == 0:
                tmp_i.append((0.0 + 1) / 2)
            else:
                tmp_i.append((((inner / (norm1 * norm2)).item()) + 1) / 2)
        cosine_similarity.append(tmp_i)

    # average cosine similarity calculate
    mean_model_cosine = []
    for i in range(len(gradients)):
        mean = 0.0
        for j in range(len(gradients)):
            mean += cosine_similarity[i][j]
        mean_model_cosine.append(mean / len(gradients))

    # aggregation weights calculate
    mean_model_cosine = torch.tensor(mean_model_cosine)
    agg_weights = mean_model_cosine / sum(mean_model_cosine)

    return agg_weights

def gl_model_output_proxy(central_node, client_params):
    # proxy data
    test_loader = central_node.validate_set_noshuffle

    # global model output with proxy data
    global_output = []
    central_node.model.cuda().eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = central_node.model(data)
            for i in range(len(output)):
                global_output.append(output[i])

    # local model output with proxy data
    local_output = []
    for i in range(len(client_params)):
        model = copy.deepcopy(central_node.model)
        model.load_state_dict(client_params[i])
        model.cuda().eval()
        tmp = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                for j in range(len(output)):
                    tmp.append(output[j])
        local_output.append(tmp)

    # global-local output distance calculate
    output_distance = []
    for i in range(len(client_params)):
        tmp = 0.0
        for j in range(len(global_output)):
            tmp += torch.linalg.vector_norm(global_output[j] - local_output[i][j], ord=2)
        output_distance.append((tmp / len(global_output)).item())

    # aggregation weights calculate
    output_distance = torch.tensor(output_distance)
    agg_weights = (1 / output_distance) / sum(1 / output_distance)

    return agg_weights

def gl_model_output_snd(central_node, client_params):
    # standard normal distribution noise
    test_loader = central_node.validate_set_noshuffle
    snd = torch.normal(0, 1, size=(len(test_loader.dataset), 3, 32, 32))

    # global model output with standard normal distribution noise
    global_output = []
    central_node.model.cuda().eval()
    with torch.no_grad():
        data = snd.cuda()
        output = central_node.model(data)
        for i in range(len(output)):
            global_output.append(output[i])

    # local model output with standard normal distribution noise
    local_output = []
    for i in range(len(client_params)):
        model = copy.deepcopy(central_node.model)
        model.load_state_dict(client_params[i])
        model.cuda().eval()
        tmp = []
        with torch.no_grad():
            data = snd.cuda()
            output = model(data)
            for j in range(len(output)):
                tmp.append(output[j])
        local_output.append(tmp)

    # global-local output distance calculate
    output_distance = []
    for i in range(len(client_params)):
        tmp = 0.0
        for j in range(len(global_output)):
            tmp += torch.linalg.vector_norm(global_output[j] - local_output[i][j], ord=2)
        output_distance.append((tmp / len(global_output)).item())

    # aggregation weights calculate
    output_distance = torch.tensor(output_distance)
    agg_weights = (1 / output_distance) / sum(1 / output_distance)

    return agg_weights

def ll_model_output_proxy(central_node, client_params):
    # proxy data
    test_loader = central_node.validate_set_noshuffle

    # local model output with proxy data
    local_output = []
    for i in range(len(client_params)):
        model = copy.deepcopy(central_node.model)
        model.load_state_dict(client_params[i])
        model.cuda().eval()
        tmp = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                for j in range(len(output)):
                    tmp.append(output[j])
        local_output.append(tmp)

    # local-local output distance calculate
    output_distance = []
    for i in range(len(client_params)):
        tmp_i = 0.0
        for j in range(len(client_params)):
            tmp_j = 0.0
            for k in range(len(local_output)):
                tmp_j += torch.linalg.vector_norm(local_output[i][k] - local_output[j][k], ord=2)
            tmp_j = (tmp_j / len(local_output)).item()
            tmp_i += tmp_j
        output_distance.append(tmp_i / len(client_params))

    # aggregation weights calculate
    output_distance = torch.tensor(output_distance)
    agg_weights = (1 / output_distance) / sum(1 / output_distance)

    return agg_weights

def ll_model_output_snd(central_node, client_params):
    # standard normal distribution noise
    test_loader = central_node.validate_set_noshuffle
    snd = torch.normal(0, 1, size=(len(test_loader.dataset), 3, 32, 32))

    # local model output with standard normal distribution noise
    local_output = []
    for i in range(len(client_params)):
        model = copy.deepcopy(central_node.model)
        model.load_state_dict(client_params[i])
        model.cuda().eval()
        tmp = []
        with torch.no_grad():
            data = snd.cuda()
            output = model(data)
            for j in range(len(output)):
                tmp.append(output[j])
        local_output.append(tmp)

    # local-local output distance calculate
    output_distance = []
    for i in range(len(client_params)):
        tmp_i = 0.0
        for j in range(len(client_params)):
            tmp_j = 0.0
            for k in range(len(local_output)):
                tmp_j += torch.linalg.vector_norm(local_output[i][k] - local_output[j][k], ord=2)
            tmp_j = (tmp_j / len(local_output)).item()
            tmp_i += tmp_j
        output_distance.append(tmp_i / len(client_params))

    # aggregation weights calculate
    output_distance = torch.tensor(output_distance)
    agg_weights = (1 / output_distance) / sum(1 / output_distance)

    return agg_weights

def model_acc_proxy(central_node, client_params):
    # proxy data
    test_loader = central_node.validate_set_noshuffle

    # local model acc with proxy data
    local_acc = []
    for i in range(len(client_params)):
        model = copy.deepcopy(central_node.model)
        model.load_state_dict(client_params[i])
        model.cuda().eval()
        correct = 0.0
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(test_loader.dataset)
        local_acc.append(acc)

    # aggregation weights calculate
    local_acc = torch.tensor(local_acc)
    agg_weights = local_acc / sum(local_acc)

    return agg_weights

def model_class(central_node, client_params, data, select_list):
    # calculate number of class per client
    class_per_client = []
    np.set_printoptions(precision=6, suppress=True)
    for i in select_list:
        num_class = 0
        for j in range(len(data.proportion[i])):
            if data.proportion[i][j] > 0:
                num_class += 1
        class_per_client.append(num_class)
    print(class_per_client)

    # calculate aggregation weights
    class_per_client = torch.tensor(class_per_client)
    agg_weights = class_per_client / sum(class_per_client)
    print(agg_weights)
    return agg_weights

##############################################################################
# fedlaw function
##############################################################################

def fedlaw_generate_global_model(gamma, optmized_weights, client_params, central_node):
    for i in range(len(client_params)):
        if i == 0:
            fedlaw_param = gamma*optmized_weights[i]*client_params[i]
        else:
            fedlaw_param = fedlaw_param.add(gamma*optmized_weights[i]*client_params[i])
    central_node.model.load_param(copy.deepcopy(fedlaw_param.detach()))
    
    return central_node

## FedLAW (for SWA: first lambdas then gamma)
def fedlaw_optimization(args, size_weights, parameters, central_node):
    '''
    fedlaw optimization functions for optimize both gamma and lambdas
    '''
    if args.dataset == 'cifar10':
        server_lr = 0.01
    else:
        server_lr = 0.005

    cohort_size = len(parameters)

    if args.whether_swa == 'none':
        # initialize gamma and lambdas
        # the last element is gamma
        if args.server_funct == 'exp':
            optimizees = torch.tensor([torch.log(torch.tensor(j)) for j in size_weights] + [0.0], device='cuda', requires_grad=True)
        elif args.server_funct == 'quad':
            optimizees = torch.tensor([math.sqrt(1.0/cohort_size) for j in size_weights]+ [1.0], device='cuda', requires_grad=True)
        optimizee_list = [optimizees]

        if args.server_optimizer == 'adam':
            optimizer = optim.Adam(optimizee_list, lr=server_lr, betas=(0.5, 0.999))
        elif args.server_optimizer == 'sgd':
            optimizer = optim.SGD(optimizee_list, lr=server_lr, momentum=0.9)
        else:
            raise ValueError('fusion optimizer is not defined!')

        # set the scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                        gamma=0.5)

        # clear grad
        for i in range(len(optimizee_list)):
            optimizee_list[i].grad = torch.zeros_like(optimizee_list[i])

        # Train optimizees
        softmax = nn.Softmax(dim=0)
        # set the model as train to update the buffers for normalization layers
        central_node.model.train()
        for epoch in range(args.server_epochs): 
            # the training data is the small dataset on the server
            train_loader = central_node.validate_set
            for itr, (data, target) in enumerate(train_loader):
                for i in range(cohort_size):
                    if i == 0:
                        if args.server_funct == 'exp':
                            model_param = torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i]*parameters[i]
                        elif args.server_funct == 'quad':
                            model_param = optimizees[-1]*optimizees[-1]*optimizees[i]*optimizees[i]/sum(optimizees[:-1]*optimizees[:-1])*parameters[i]
                    else:
                        if args.server_funct == 'exp':
                            model_param = model_param.add(torch.exp(optimizees[-1])*softmax(optimizees[:-1])[i]*parameters[i])
                        elif args.server_funct == 'quad':
                            model_param = model_param.add(optimizees[-1]*optimizees[-1]*optimizees[i]*optimizees[i]/sum(optimizees[:-1]*optimizees[:-1])*parameters[i])

                # train model
                data, target = data.cuda(), target.cuda()

                # Update optimizees
                # zero_grad
                optimizer.zero_grad()
                # update models according to the lr
                output = central_node.model.forward_with_param(data, model_param)
                loss =  F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            # scheduling
            scheduler.step()
        # record and print current lam
        if args.server_funct == 'exp':
            optmized_weights = [j for j in softmax(optimizees[:-1]).detach().cpu().numpy()]
            learned_gamma = torch.exp(optimizees[-1])
        elif args.server_funct == 'quad':
            optmized_weights = [j*j/sum(optimizees[:-1]*optimizees[:-1]) for j in optimizees[:-1].detach().cpu().numpy()]
            learned_gamma = optimizees[-1]*optimizees[-1]

    elif args.whether_swa == 'swa':
        # Two stage strategy: first, train lambdas; second, train gamma

        ## Optimize Lambdas ##
        # initialize fusion weights
        optimizees = []
        if args.server_funct == 'exp':
            lam = torch.tensor([torch.log(torch.tensor(j)) for j in size_weights], device='cuda', requires_grad=True)
        elif args.server_funct == 'quad':
            lam = torch.tensor([math.sqrt(1.0/cohort_size) for j in size_weights], device='cuda', requires_grad=True)
        optimizees.append(lam)

        # set the optimizer
        if args.server_optimizer == 'adam':
            optimizer = optim.Adam(optimizees, lr=server_lr, betas=(0.5, 0.999))
        elif args.server_optimizer == 'sgd':
            optimizer = optim.SGD(optimizees, lr=server_lr, momentum=0.9)
        else:
            raise ValueError('fusion optimizer is not defined!')

        # set the scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        swa_model = AveragedModel(lam)
        swa_start = 5
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

        # clear grad
        for i in range(len(optimizees)):
            optimizees[i].grad = torch.zeros_like(optimizees[i])

        # train optimizees
        softmax = nn.Softmax(dim=0)
        central_node.model.train()
        for epoch in range(args.server_epochs//2): 
            # the training data is the small dataset on the server
            train_loader = central_node.validate_set 
            for _, (data, target) in enumerate(train_loader):
                for i in range(cohort_size):
                    if i == 0:
                        if args.server_funct == 'exp':
                            model_param = softmax(lam)[i]*parameters[i]
                        elif args.server_funct == 'quad':
                            model_param = lam[i]*lam[i]/sum(lam*lam)*parameters[i]
                        # print(learned_gamma)
                    else:
                        if args.server_funct == 'exp':
                            model_param = model_param.add(softmax(lam)[i]*parameters[i])
                        elif args.server_funct == 'quad':
                            model_param = model_param.add(lam[i]*lam[i]/sum(lam*lam)*parameters[i])
                # train model
                data, target = data.cuda(), target.cuda()

                # update optimizees
                # zero_grad
                optimizer.zero_grad()
                # update models according to the lr
                output = central_node.model.forward_with_param(data, model_param)
                loss =  F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            # scheduling
            if epoch > swa_start:
                swa_model.update_parameters(lam)
                swa_scheduler.step()
            else:
                scheduler.step()
        if args.server_funct == 'exp':
            optmized_weights = [j for j in softmax(swa_model.module).detach().cpu().numpy()]
        elif args.server_funct == 'quad':
            optmized_weights = [j*j/sum(swa_model.module*swa_model.module).detach().cpu().numpy() for j in swa_model.module.detach().cpu().numpy()]

        ## Train gamma ##
        # initialize fusion weights
        optimizees = []
        if args.server_funct == 'exp':
            gamma = torch.tensor(0.0, device='cuda', requires_grad=True)
        elif args.server_funct == 'quad':
            gamma = torch.tensor(1.0, device='cuda', requires_grad=True)
        optimizees.append(gamma)

        # set the optimizer
        if args.server_optimizer == 'adam':
            optimizer = optim.Adam(optimizees, lr=server_lr, betas=(0.5, 0.999))
        elif args.server_optimizer == 'sgd':
            optimizer = optim.SGD(optimizees, lr=server_lr, momentum=0.9)
        else:
            raise ValueError('fusion optimizer is not defined!')
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                        gamma=0.5)

        # clear grad
        for i in range(len(optimizees)):
            optimizees[i].grad = torch.zeros_like(optimizees[i])

        # train optimizees
        central_node.model.train()
        for epoch in range(args.server_epochs//2): 
            # the training data is the small dataset on the server
            train_loader = central_node.validate_set 

            for _, (data, target) in enumerate(train_loader):

                for i in range(cohort_size):
                    if i == 0:
                        if args.server_funct == 'exp':
                            model_param = torch.exp(optimizees[-1])*optmized_weights[i]*parameters[i]
                        elif args.server_funct == 'quad':
                            model_param = optimizees[-1]*optimizees[-1]*optmized_weights[i]*parameters[i]
                    else:
                        if args.server_funct == 'exp':
                            model_param = model_param.add(torch.exp(optimizees[-1])*optmized_weights[i]*parameters[i])
                        elif args.server_funct == 'quad':
                            model_param = model_param.add(optimizees[-1]*optimizees[-1]*optmized_weights[i]*parameters[i])

                # train model
                data, target = data.cuda(), target.cuda()

                # update optimizees
                # zero_grad
                optimizer.zero_grad()
                # update models according to the lr
                output = central_node.model.forward_with_param(data, model_param)
                loss =  F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        if args.server_funct == 'exp':
            learned_gamma = copy.deepcopy(torch.exp(optimizees[-1]).detach())
        elif args.server_funct == 'quad':
            learned_gamma = copy.deepcopy((optimizees[-1]*optimizees[-1]).detach())

    return learned_gamma, optmized_weights


##############################################################################
# Baselines function (FedAvg, FedDF, FedBE, FedDyn, FedAdam, Finetune, etc.)
##############################################################################

def Server_update(args, central_node, client_nodes, select_list, size_weights):
    '''
    server update functions for baselines
    '''

    # receive the local models from clients
    agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)

    # update the global model
    if args.server_method == 'fedavg':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        if args.client_method == 'scaffold':
            avg_c = []
            for param in central_node.model.parameters():
                avg_c.append(torch.zeros_like(param))
            for idx in select_list:
                for i in range(len(avg_c)):
                    avg_c[i] += (client_nodes[idx].c[i] - client_nodes[idx].previous_c[i]) / len(client_nodes)
            # avg_c = avg_c / len(client_nodes)
            central_node.previous_c = copy.deepcopy(central_node.c)
            for i in range(len(avg_c)):
                central_node.c[i] = central_node.c[i] + avg_c[i]

    elif args.server_method == 'feddf':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        central_node = feddf(args, central_node, client_nodes, select_list)

    elif args.server_method == 'fedbe':
        prev_global_param = copy.deepcopy(central_node.model.state_dict())
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        central_node = fedbe(args, prev_global_param, central_node, client_nodes, select_list)

    elif args.server_method == 'finetune':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        central_node = server_finetune(args, central_node)

    elif args.server_method == 'feddyn':
        central_node = feddyn(args, central_node, agg_weights, client_nodes, select_list)
    
    elif args.server_method == 'fedadam':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node = fedadam(args, central_node, avg_global_param)

    else:
        raise ValueError('Undefined server method...')

    return central_node

# FedAvg
def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params

# Sever Finetune
def server_finetune(args, central_node):
    central_node.model.train()
    for epoch in range(args.server_epochs): 
        # the training data is the small dataset on the server
        train_loader = central_node.validate_set 

        for _, (data, target) in enumerate(train_loader):

            central_node.optimizer.zero_grad()
            # train model
            data, target = data.cuda(), target.cuda()

            output = central_node.model(data)

            # compute losses according to the weights
            loss =  F.cross_entropy(output, target)
            loss.backward()
            central_node.optimizer.step()

    return central_node

# FedDF
def divergence(student_logits, teacher_logits):
    divergence = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )  # forward KL
    return divergence

def feddf(args, central_node, client_nodes, select_list):
    # train and update
    central_node.model.cuda().train()
    nets = []
    for client_idx in select_list:
        client_nodes[client_idx].model.cuda().eval()
        nets.append(client_nodes[client_idx].model)

    for _ in range(args.server_epochs):
        train_loader = central_node.validate_set 

        for _, (data, target) in enumerate(train_loader):
            central_node.optimizer.zero_grad()
            # train model
            data, target = data.cuda(), target.cuda()

            output = central_node.model(data)
            teacher_logits = sum([net(data).detach() for net in nets]) / len(select_list)

            loss = divergence(output, teacher_logits)
            loss.backward()
            central_node.optimizer.step()

    return central_node

# FedBE
class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter and the number of models already averaged; if None,
            equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with `avg_fn` parameter.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights.

    Example:
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            0.1 * averaged_model_parameter + 0.9 * model_parameter
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg, use_buffers=True)

    .. note::
        When using SWA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.

    .. note::
        :attr:`avg_fn` is not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    """
    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (self.module
        )
        model_param = (model
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1

@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class SWALR(_LRScheduler):
    r"""Anneals the learning rate in each parameter group to a fixed value.

    This learning rate scheduler is meant to be used with Stochastic Weight
    Averaging (SWA) method (see `torch.optim.swa_utils.AveragedModel`).

    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lrs (float or list): the learning rate value for all param groups
            together or separately for each group.
        annealing_epochs (int): number of epochs in the annealing phase
            (default: 10)
        annealing_strategy (str): "cos" or "linear"; specifies the annealing
            strategy: "cos" for cosine annealing, "linear" for linear annealing
            (default: "cos")
        last_epoch (int): the index of the last epoch (default: -1)

    The :class:`SWALR` scheduler is can be used together with other
    schedulers to switch to a constant learning rate late in the training
    as in the example below.

    Example:
        >>> loader, optimizer, model = ...
        >>> lr_lambda = lambda epoch: 0.9
        >>> scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
        >>>        lr_lambda=lr_lambda)
        >>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
        >>>        anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
        >>> swa_start = 160
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    """
    def __init__(self, optimizer, swa_lr, anneal_epochs=10, anneal_strategy='cos', last_epoch=-1):
        swa_lrs = self._format_param(optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups):
            group['swa_lr'] = swa_lr
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', "
                             f"instead got {anneal_strategy}")
        elif anneal_strategy == 'cos':
            self.anneal_func = self._cosine_anneal
        elif anneal_strategy == 'linear':
            self.anneal_func = self._linear_anneal
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}")
        self.anneal_epochs = anneal_epochs
        super(SWALR, self).__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(optimizer, swa_lrs):
        if isinstance(swa_lrs, (list, tuple)):
            if len(swa_lrs) != len(optimizer.param_groups):
                raise ValueError("swa_lr must have the same length as "
                                 f"optimizer.param_groups: swa_lr has {len(swa_lrs)}, "
                                 f"optimizer.param_groups has {len(optimizer.param_groups)}")
            return swa_lrs
        else:
            return [swa_lrs] * len(optimizer.param_groups)

    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        step = self._step_count - 1
        if self.anneal_epochs == 0:
            step = max(1, step)
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [self._get_initial_lr(group['lr'], group['swa_lr'], prev_alpha)
                    for group in self.optimizer.param_groups]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [group['swa_lr'] * alpha + lr * (1 - alpha)
                for group, lr in zip(self.optimizer.param_groups, prev_lrs)]



class SWAG_server(torch.nn.Module):
    def __init__(self, base_model, avg_model=None, max_num_models=25, var_clamp=1e-5, concentrate_num=1):
        self.base_model = base_model
        self.max_num_models=max_num_models
        self.var_clamp=var_clamp
        self.concentrate_num = concentrate_num
        self.avg_model = avg_model
         
    def compute_var(self, mean, sq_mean): 
        var_dict = {}
        for k in mean.keys():
          var = torch.clamp(sq_mean[k] - mean[k] ** 2, self.var_clamp) 
          var_dict[k] = var 

        return var_dict

    def compute_mean_sq(self, teachers):
        w_avg = {}
        w_sq_avg = {}
        w_norm ={}
        
        for k in teachers[0].keys():
            if "batches_tracked" in k: continue
            w_avg[k] = torch.zeros(teachers[0][k].size())
            w_sq_avg[k] = torch.zeros(teachers[0][k].size())
            w_norm[k] = 0.0 
          
        for k in w_avg.keys():
            if "batches_tracked" in k: continue
            for i in range(0, len(teachers)):
              grad = teachers[i][k].cpu()- self.base_model[k].cpu()
              norm = torch.norm(grad, p=2)
              
              grad = grad/norm
              sq_grad = grad**2
              
              w_avg[k] += grad
              w_sq_avg[k] += sq_grad
              w_norm[k] += norm
              
            w_avg[k] = torch.div(w_avg[k], len(teachers))
            w_sq_avg[k] = torch.div(w_sq_avg[k], len(teachers))
            w_norm[k] = torch.div(w_norm[k], len(teachers))
            
        return w_avg, w_sq_avg, w_norm
        
    def construct_models(self, teachers, mean=None, mode="dir"):
      if mode=="gaussian":
        w_avg, w_sq_avg, w_norm= self.compute_mean_sq(teachers)
        w_var = self.compute_var(w_avg, w_sq_avg)      
        
        mean_grad = copy.deepcopy(w_avg)
        for i in range(self.concentrate_num):
          for k in w_avg.keys():
            mean = w_avg[k]
            var = torch.clamp(w_var[k], 1e-6)
            
            eps = torch.randn_like(mean)
            sample_grad = mean + torch.sqrt(var) * eps * 0.1
            mean_grad[k] = (i*mean_grad[k] + sample_grad) / (i+1)
        
        for k in w_avg.keys():
          mean_grad[k] = mean_grad[k]*1.0*w_norm[k] + self.base_model[k].cpu()
          
        return mean_grad  
      
      elif mode=="random":
        num_t = 3
        ts = np.random.choice(teachers, num_t, replace=False)
        mean_grad = {}
        for k in ts[0].keys():
          mean_grad[k] = torch.zeros(ts[0][k].size())
          for i, t in enumerate(ts):
            mean_grad[k]+= t[k]
          
        for k in ts[0].keys():
          mean_grad[k]/=num_t  
          
        return mean_grad
      
      elif mode=="dir":
        proportions = np.random.dirichlet(np.repeat(1.0, len(teachers)))
        mean_grad = {}
        for k in teachers[0].keys():
          mean_grad[k] = torch.zeros(teachers[0][k].size())
          for i, t in enumerate(teachers):
            mean_grad[k]+= t[k]*proportions[i]
          
        for k in teachers[0].keys():
          mean_grad[k]/=sum(proportions)  

        return mean_grad   

def fedbe(args, prev_global_param, central_node, client_nodes, select_list):
    # generate teachers
    nets = []
    base_teachers = []

    fedavg_model = init_model(args.local_model, args).cuda()
    swag_model = init_model(args.local_model, args).cuda()
    fedavg_model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
    nets.append(copy.deepcopy(fedavg_model))

    for client_idx in select_list:
        client_nodes[client_idx].model.cuda().eval()
        nets.append(copy.deepcopy(client_nodes[client_idx].model))
        base_teachers.append(copy.deepcopy(client_nodes[client_idx].model.state_dict()))

    # generate swag model
    swag_server = SWAG_server(prev_global_param, avg_model=copy.deepcopy(central_node.model.state_dict()), concentrate_num=1)
    w_swag = swag_server.construct_models(base_teachers, mode='gaussian') 
    swag_model.load_state_dict(w_swag)
    nets.append(copy.deepcopy(swag_model))  

    # train and update
    central_node.model.cuda().train()
    for _ in range(args.server_epochs):
        train_loader = central_node.validate_set 

        for _, (data, target) in enumerate(train_loader):
            central_node.optimizer.zero_grad()
            # train model
            data, target = data.cuda(), target.cuda()

            output = central_node.model(data)
            teacher_logits = sum([net(data).detach() for net in nets]) / len(nets)

            loss = divergence(output, teacher_logits)
            loss.backward()
            central_node.optimizer.step()

    return central_node


# FedDyn
def feddyn(args, central_node, agg_weights, client_nodes, select_list):
    '''
    server function for feddyn
    '''

    # update server's state
    uploaded_models = []
    for i in select_list:
        uploaded_models.append(copy.deepcopy(client_nodes[i].model))

    model_delta = copy.deepcopy(uploaded_models[0])
    for param in model_delta.parameters():
        param.data = torch.zeros_like(param.data)

    for idx, client_model in enumerate(uploaded_models):
        for server_param, client_param, delta_param in zip(central_node.model.parameters(), client_model.parameters(), model_delta.parameters()):
            delta_param.data += (client_param - server_param) * agg_weights[idx]

    for state_param, delta_param in zip(central_node.server_state.parameters(), model_delta.parameters()):
        state_param.data -= args.mu * delta_param

    # aggregation
    central_node.model = copy.deepcopy(uploaded_models[0])
    for param in central_node.model.parameters():
        param.data = torch.zeros_like(param.data)
        
    for idx, client_model in enumerate(uploaded_models):
        for server_param, client_param in zip(central_node.model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * agg_weights[idx]

    for server_param, state_param in zip(central_node.model.parameters(), central_node.server_state.parameters()):
        server_param.data -= (1/args.mu) * state_param

    return central_node


# FedAdam
def fedadam(args, central_node, avg_global_param):
    # hyperparam for fedadam, suggested in their paper, cifar10
    # lr_g = 0.01
    lr_g = float(args.fedadam_server_lr)
    beta1 = 0.9
    beta2 = 0.99
    w = copy.deepcopy(central_node.model)
    w.load_state_dict(avg_global_param)
    w_t = copy.deepcopy(central_node.model)

    # compute delta_w_t
    delta_w_t = copy.deepcopy(w_t)
    for delta_w_t_param, w_t_param, w_param in zip(delta_w_t.parameters(), w_t.parameters(), w.parameters()):
        delta_w_t_param.data = w_param.data - w_t_param.data

    # compute param
    for delta_w_t_param, m_param, v_param, w_t_param, w_param in zip(delta_w_t.parameters(), central_node.m.parameters(), central_node.v.parameters(), w_t.parameters(), w.parameters()):
        m_param.data = beta1*m_param.data+(1-beta1)*delta_w_t_param.data
        v_param.data = beta2*v_param.data+(1-beta2)*delta_w_t_param.data.pow(2)
        w_param.data = w_t_param.data + lr_g*m_param.data/(torch.sqrt(v_param.data)+1e-5)

    central_node.model = copy.deepcopy(w)
    return central_node