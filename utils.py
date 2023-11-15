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


##############################################################################
# Validation function
##############################################################################

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
def get_gradients(args, central_node, clients_nodes, select_list):
    # 일단은 gradients 자체만 구해서 리턴 후 저장해보자
    # 만약 크기가 너무 커서 비효율적이라면 필요한 계산까지 해서 저장해보자
    # 20 client 200 round 면 4GB 인듯 너무 큼
    # 여기서 계산까지해서 보내자
    global_model = copy.deepcopy(central_node.model.state_dict())

    # local gradients calculate
    all_gradients = []
    participation_gradients = []
    for i in range(len(clients_nodes)):
        param = copy.deepcopy(global_model)
        client_param = copy.deepcopy(clients_nodes[i].model.state_dict())
        for name_param in param:
            param[name_param] = global_model[name_param] - client_param[name_param]
        # get gradients of all clients
        all_gradients.append(copy.deepcopy(param))
        # # get gradients of participation clients
        # if i in select_list:
        #     participation_gradients.append(copy.deepcopy(param))

    # gradients concatenate
    all_flatted_gradients = []
    participation_flatted_gradients = []
    for i in range(len(all_gradients)):
        tmp = None
        for name_param in all_gradients[i]:
            if tmp == None:
                tmp = torch.flatten(copy.deepcopy(all_gradients[i][name_param]))
            else:
                tmp = torch.cat((tmp, torch.flatten(copy.deepcopy(all_gradients[i][name_param]))))
        all_flatted_gradients.append(tmp)
        # if i in select_list:
        #     participation_flatted_gradients.append(tmp)

    # calculate gl model-wise distance
    all_gl_model_distance = []
    participation_gl_model_distance = []
    for i in range(len(all_flatted_gradients)):
        distance = torch.linalg.vector_norm(all_flatted_gradients[i], ord=2).item()
        distance = np.array(distance)
        all_gl_model_distance.append(distance)
        if i in select_list:
            participation_gl_model_distance.append(distance)

    # calculate gl layer-wise distance
    all_gl_layer_distance = []
    participation_gl_layer_distance = []
    for i in range(len(all_gradients)):
        tmp = []
        for name_param in all_gradients[i]:
            layer_distance = torch.linalg.vector_norm(all_gradients[i][name_param], ord=2).item()
            tmp.append(layer_distance)
        tmp = np.average(tmp)
        all_gl_layer_distance.append(tmp)
        if i in select_list:
            participation_gl_layer_distance.append(tmp)

    # calculate ll model-wise distance
    all_ll_model_distance = []
    participation_ll_model_distance = []
    for i in range(len(all_flatted_gradients)):
        tmp = []
        participation_tmp = []
        for j in range(len(all_flatted_gradients)):
            all_model_distance = torch.linalg.vector_norm(all_flatted_gradients[i] - all_flatted_gradients[j],
                                                          ord=2).item()
            tmp.append(all_model_distance)
            if j in select_list:
                participation_model_distance = torch.linalg.vector_norm(
                    all_flatted_gradients[i] - all_flatted_gradients[j], ord=2).item()
                participation_tmp.append(participation_model_distance)
        tmp = np.array(tmp)
        all_ll_model_distance.append(tmp)
        if i in select_list:
            participation_tmp = np.array(participation_tmp)
            participation_ll_model_distance.append(participation_tmp)

    # calculate ll layer-wise distance
    all_ll_layer_distance = []
    participation_ll_layer_distance = []
    for i in range(len(all_gradients)):
        tmp = []
        part_tmp = []
        for j in range(len(all_gradients)):
            tmp_layer = []
            part_tmp_layer = []
            for name_param in all_gradients[i]:
                layer_distance = torch.linalg.vector_norm(all_gradients[i][name_param] - all_gradients[j][name_param],
                                                          ord=2).item()
                tmp_layer.append(layer_distance)
                if j in select_list:
                    part_layer_distance = torch.linalg.vector_norm(
                        all_gradients[i][name_param] - all_gradients[j][name_param], ord=2).item()
                    part_tmp_layer.append(part_layer_distance)
            tmp_layer = np.average(tmp_layer)
            tmp.append(tmp_layer)
            if j in select_list:
                part_tmp_layer = np.average(part_tmp_layer)
                part_tmp.append(part_tmp_layer)
        all_ll_layer_distance.append(tmp)
        if i in select_list:
            participation_ll_layer_distance.append(part_tmp)

    all_distance = {'gl_model' : np.array(all_gl_model_distance), 'gl_layer' : np.array(all_gl_layer_distance),
                    'll_model' : np.array(all_ll_model_distance), 'll_layer' : np.array(all_ll_layer_distance)}
    participation_distance = {'gl_model': np.array(participation_gl_model_distance), 'gl_layer': np.array(participation_gl_layer_distance),
                    'll_model': np.array(participation_ll_model_distance), 'll_layer': np.array(participation_ll_layer_distance)}
    return all_distance, participation_distance

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


