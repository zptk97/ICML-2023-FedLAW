import copy

import numpy
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.backends import cudnn
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn


##############################################################################
# Tools
##############################################################################

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


def model_parameter_vector(args, model):
    if 'fedlaw' in args.server_method:
        vector = model.flat_w
    else:
        param = [p.view(-1) for p in model.parameters()]
        vector = torch.cat(param, dim=0)
    return vector


##############################################################################
# Initialization function
##############################################################################

def init_model(model_type, args):
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100

    if 'fedlaw' in args.server_method:
        if model_type == 'CNN':
            if args.dataset == 'cifar10':
                model = cnn.CNNCifar10_fedlaw()
            else:
                model = cnn.CNNCifar100_fedlaw()
        elif model_type == 'ResNet20':
            model = resnet.ResNet20_fedlaw(num_classes)
        elif model_type == 'ResNet56':
            model = resnet.ResNet56_fedlaw(num_classes)
        elif model_type == 'ResNet110':
            model = resnet.ResNet110_fedlaw(num_classes)
        elif model_type == 'WRN56_2':
            model = resnet.WRN56_2_fedlaw(num_classes)
        elif model_type == 'WRN56_4':
            model = resnet.WRN56_4_fedlaw(num_classes)
        elif model_type == 'WRN56_8':
            model = resnet.WRN56_8_fedlaw(num_classes)
        elif model_type == 'DenseNet121':
            model = densenet.DenseNet121_fedlaw(num_classes)
        elif model_type == 'DenseNet169':
            model = densenet.DenseNet169_fedlaw(num_classes)
        elif model_type == 'DenseNet201':
            model = densenet.DenseNet201_fedlaw(num_classes)
        elif model_type == 'MLP':
            model = cnn.MLP_fedlaw()
        elif model_type == 'LeNet5':
            model = cnn.LeNet5_fedlaw()
    else:
        if model_type == 'CNN':
            if args.dataset == 'cifar10':
                model = cnn.CNNCifar10()
            else:
                model = cnn.CNNCifar100()
        elif model_type == 'ResNet20':
            model = resnet.ResNet20(num_classes)
        elif model_type == 'ResNet56':
            model = resnet.ResNet56(num_classes)
        elif model_type == 'ResNet110':
            model = resnet.ResNet110(num_classes)
        elif model_type == 'WRN56_2':
            model = resnet.WRN56_2(num_classes)
        elif model_type == 'WRN56_4':
            model = resnet.WRN56_4(num_classes)
        elif model_type == 'WRN56_8':
            model = resnet.WRN56_8(num_classes)
        elif model_type == 'DenseNet121':
            model = densenet.DenseNet121(num_classes)
        elif model_type == 'DenseNet169':
            model = densenet.DenseNet169(num_classes)
        elif model_type == 'DenseNet201':
            model = densenet.DenseNet201(num_classes)
        elif model_type == 'MLP':
            model = cnn.MLP()
        elif model_type == 'LeNet5':
            model = cnn.LeNet5()

    return model


def init_optimizer(num_id, model, args):
    optimizer = []
    if num_id > -1 and args.client_method == 'fedprox':
        optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
    elif num_id > -1 and args.client_method == 'scaffold':
        optimizer = SCAFFOLDOptimizer(model.parameters(), lr=args.lr)
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
    return optimizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


##############################################################################
# Training function
##############################################################################

def generate_selectlist(client_node, ratio=0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
    return select_list


def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99  # 0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # g = g.cuda()
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])

##############################################################################
# Validation function
##############################################################################

def validate_withloss(args, node, which_dataset='validate'):
    node.model.cuda().eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model(data)
            # acc
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # loss
            loss_local = F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
        acc = correct / len(test_loader.dataset) * 100
        loss_value = sum(loss) / len(loss)
    return acc, loss_value


def validate(args, node, which_dataset='validate'):
    node.model.cuda().eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc


def testloss(args, node, which_dataset='validate'):
    node.model.cuda().eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model(data)
            loss_local = F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss) / len(loss)
    return loss_value


# Functions for FedLAW with param as an input
def validate_with_param(args, node, param, which_dataset='validate'):
    node.model.cuda().eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model.forward_with_param(data, param)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc


def testloss_with_param(args, node, param, which_dataset='validate'):
    node.model.cuda().eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model.forward_with_param(data, param)
            loss_local = F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss) / len(loss)
    return loss_value


##############################################################################
# Analysis function
##############################################################################

# get gradients of each client
def get_gradients(args, central_node, client_params, select_list, fedavg_agg_weights):
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    gradients = []
    for i in range(len(client_params)):
        param = copy.deepcopy(global_model)
        for name_param in param:
            param[name_param] = global_model[name_param] - client_params[i][name_param]
        gradients.append(copy.deepcopy(param))

    # FedAvg AW averaged local gradients
    for i in range(len(gradients)):
        if i == 0:
            param = copy.deepcopy(gradients[i])
            for name_param in param:
                param[name_param] = param[name_param] * fedavg_agg_weights[i]
        else:
            for name_param in param:
                param[name_param] = param[name_param] + (gradients[i][name_param] * fedavg_agg_weights[i])
        # for name_param in param:
        #     param[name_param] = param[name_param] / len(gradients)
    avg_gradients = copy.deepcopy(param)

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
    tmp = None
    for name_param in avg_gradients:
        if tmp == None:
            tmp = torch.flatten(copy.deepcopy(avg_gradients[name_param]))
        else:
            tmp = torch.cat((tmp, torch.flatten(copy.deepcopy(avg_gradients[name_param]))))
    flatted_avg_gradients = copy.deepcopy(tmp)

    # GL model-wise gradients distance calculate
    gl_model_gradients_distance = []
    for i in range(len(gradients)):
        distance = torch.linalg.vector_norm(flatted_gradients[i], ord=2).item()
        distance = np.array(distance)
        gl_model_gradients_distance.append(distance)

    # GL layer-wise gradients distance calculate
    gl_layer_gradients_distance = []
    for i in range(len(gradients)):
        tmp = []
        for name_param in gradients[i]:
            layer_distance = torch.linalg.vector_norm(gradients[i][name_param], ord=2).item()
            tmp.append(layer_distance)
        tmp = np.average(tmp)
        gl_layer_gradients_distance.append(tmp)

    # LL model-wise gradients distance calculate
    ll_model_gradients_distance = []
    for i in range(len(gradients)):
        distance = torch.linalg.vector_norm(flatted_avg_gradients - flatted_gradients[i], ord=2).item()
        distance = np.array(distance)
        ll_model_gradients_distance.append(distance)

    # LL layer-wise gradients distance calculate
    ll_layer_gradients_distance = []
    for i in range(len(gradients)):
        tmp = []
        for name_param in gradients[i]:
            layer_distance = torch.linalg.vector_norm(avg_gradients[name_param] - gradients[i][name_param], ord=2).item()
            tmp.append(layer_distance)
        tmp = np.average(tmp)
        ll_layer_gradients_distance.append(tmp)

    # save in dict
    gradients_distance = {'gl_model': np.array(gl_model_gradients_distance), 'gl_layer': np.array(gl_layer_gradients_distance),
                          'll_model': np.array(ll_model_gradients_distance), 'll_layer': np.array(ll_layer_gradients_distance)}
    return gradients_distance

def get_proportion_data(args, agg_weights, select_list, data):
    np.set_printoptions(precision=6, suppress=True)
    proportion = data.proportion
    num_classes = len(data.proportion[0])
    agg_weights = np.array(agg_weights)

    proportion_data = [0.0] * num_classes
    proportion_data = np.array(proportion_data)
    for i, idx in enumerate(select_list):
        if "layer" in args.server_method:
            tmp = proportion[idx] * np.average(agg_weights[i])
        else:
            tmp = proportion[idx] * agg_weights[i]
        proportion_data += tmp

    proportion_data = proportion_data / sum(proportion_data)

    return proportion_data


def agg_weights_fusion(args, fedavg_agg_weights, propose_agg_weights, rounds):
    if "layer" in args.server_method:
        for i in range(len(propose_agg_weights)):
            for j in range(len(propose_agg_weights[i])):
                if args.fusion == 1:
                    propose_agg_weights[i][j] = propose_agg_weights[i][j] * fedavg_agg_weights[i]
                elif args.fusion == 2:
                    # propose_agg_weights[i][j] = propose_agg_weights[i][j] + (propose_agg_weights[i][j] * fedavg_agg_weights[i])
                    propose_agg_weights[i][j] = propose_agg_weights[i][j] + (
                                1.0 * fedavg_agg_weights[i])
                    # propose_agg_weights[i][j] = ((rounds / (args.T - 1)) * propose_agg_weights[i][j]) + (((args.T - 1 - rounds) / (args.T - 1)) * fedavg_agg_weights[i])
                    # propose_agg_weights[i][j] = (propose_agg_weights[i][j]) + (
                    #             ((args.T - 1 - rounds) / (args.T - 1)) * fedavg_agg_weights[i])
                else:
                    ValueError('Undefined fusion method...')
        for i in range(len(propose_agg_weights[0])):
            sum_tmp = 0.0
            for j in range(len(propose_agg_weights)):
                sum_tmp += propose_agg_weights[j][i]
            for j in range(len(propose_agg_weights)):
                propose_agg_weights[j][i] = propose_agg_weights[j][i] / sum_tmp
        agg_weights = propose_agg_weights
    elif "model" in args.server_method:
        if args.fusion == 1:
            agg_weights = torch.tensor(fedavg_agg_weights) * propose_agg_weights
        elif args.fusion == 2:
            # agg_weights = propose_agg_weights + (torch.tensor(fedavg_agg_weights) * propose_agg_weights)
            agg_weights = propose_agg_weights + (torch.tensor(fedavg_agg_weights) * 1.0)
            # agg_weights = ((rounds / (args.T - 1)) * propose_agg_weights) + (torch.tensor(fedavg_agg_weights) * ((args.T - 1 - rounds) / (args.T - 1)))
            # agg_weights = (propose_agg_weights) + (
            #             torch.tensor(fedavg_agg_weights) * ((args.T - 1 - rounds) / (args.T - 1)))
        else:
            ValueError('Undefined fusion method...')
        agg_weights = agg_weights / sum(agg_weights)
    elif args.server_method == 'fedavg':
        if args.fusion == 1:
            agg_weights = torch.tensor(fedavg_agg_weights) * propose_agg_weights
        elif args.fusion == 2:
            # agg_weights = propose_agg_weights + (torch.tensor(fedavg_agg_weights) * propose_agg_weights)
            agg_weights = propose_agg_weights + (torch.tensor(fedavg_agg_weights) * 1.0)
            # agg_weights = ((rounds / (args.T - 1)) * propose_agg_weights) + (
            #             torch.tensor(fedavg_agg_weights) * ((args.T - 1 - rounds) / (args.T - 1)))
            # agg_weights = (propose_agg_weights) + (
            #         torch.tensor(fedavg_agg_weights) * ((args.T - 1 - rounds) / (args.T - 1)))
        else:
            ValueError('Undefined fusion method...')
        agg_weights = agg_weights / sum(agg_weights)
    else:
        raise ValueError('Undefined server method...')
    return agg_weights


def agg_weights_scale(args, fedavg_agg_weights, propose_agg_weights, select_list, data, rounds):
    # Calculate FedAvg effective training data
    proportion = data.proportion
    fedavg_agg_weights = np.array(fedavg_agg_weights)
    sum_fedavg = 0.0
    for i, idx in enumerate(select_list):
        sum_fedavg += sum(proportion[idx]) * fedavg_agg_weights[i]
    # Calculate Propose effective training data
    agg_weights = np.zeros_like(fedavg_agg_weights)
    if "layer" in args.server_method:
        for i in range(len(propose_agg_weights)):
            for j in range(len(propose_agg_weights[i])):
                agg_weights[i] += propose_agg_weights[i][j]
            agg_weights[i] = agg_weights[i] / len(propose_agg_weights[i])
    elif "model" in args.server_method:
        agg_weights = np.array(propose_agg_weights)
    elif args.server_method == 'fedavg':
        agg_weights = np.array(propose_agg_weights)
    else:
        raise ValueError('Undefined server method...')
    sum_propose = 0.0
    for i, idx in enumerate(select_list):
        sum_propose += sum(proportion[idx]) * agg_weights[i]
    # get scaling factor
    scaling_factor = sum_fedavg / sum_propose
    # get gamma
    if scaling_factor > 1.0:
        scaling_factor = scaling_factor - 1.0
        scaling_factor = scaling_factor * np.exp(-1.0 * rounds)
        scaling_factor = scaling_factor + 1.0
    else:
        scaling_factor = 1.0

    return scaling_factor


def get_global_weights(central_node):
    # get global model's flatted weights
    global_model = copy.deepcopy(central_node.model.state_dict())
    # gradients concatenate
    flatted_weights = []
    for name_param in global_model:
        if len(flatted_weights) == 0:
            flatted_weights = torch.flatten(global_model[name_param])
        else:
            flatted_weights = torch.cat((flatted_weights, torch.flatten(global_model[name_param])))
    flatted_weights = flatted_weights.to('cpu')
    return flatted_weights

