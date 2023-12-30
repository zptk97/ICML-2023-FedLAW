import numpy as np
import torch
import torch.nn.functional as F
from utils import validate, model_parameter_vector
import copy
from nodes import Node

##############################################################################
# General client function 
##############################################################################

def receive_server_model(args, client_nodes, central_node):

    for idx in range(len(client_nodes)):
        if 'fedlaw' in args.server_method:
            client_nodes[idx].model.load_param(copy.deepcopy(central_node.model.get_param(clone = True)))
        else:
            client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

    return client_nodes

def Client_update(args, client_nodes, central_node):
    '''
    client update functions
    '''
    # 지금은 모든 client에 distribute하고 모든 client에서 학습 중
    # client 수가 많아지면 분명 학습 시간에 큰 문제가 있을 것
    # 실험을 위해 이렇게 한 것 같은데, 추후 빠른 버전을 만들던가 해야할 듯
    # clients receive the server model 
    client_nodes = receive_server_model(args, client_nodes, central_node)

    # update the global model
    if args.client_method == 'local_train':
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'scaffold':
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_scaffold(args, client_nodes[i], central_node)
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses) / len(epoch_losses))
            train_loss = sum(client_losses) / len(client_losses)
            # update c of each client, ci
            client_nodes[i].previous_c = copy.deepcopy(client_nodes[i].c)
            for ci, c, x, yi in zip(client_nodes[i].c, central_node.c, central_node.model.parameters(),
                                    client_nodes[i].model.parameters()):
                ci.data = ci - c + 1 / len(client_nodes[i].local_data) / args.lr * (x - yi)

    elif args.client_method == 'fedprox':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedprox(global_model_param, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'feddyn':
        global_model_vector = copy.deepcopy(model_parameter_vector(args, central_node.model).detach().clone())
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_feddyn(global_model_vector, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

            # update old grad
            v1 = model_parameter_vector(args, client_nodes[i].model).detach()
            client_nodes[i].old_grad = client_nodes[i].old_grad - args.mu * (v1 - global_model_vector)

    else:
        raise ValueError('Undefined server method...')

    return client_nodes, train_loss

def Client_validate(args, client_nodes):
    '''
    client validation functions, for testing local personalization
    '''
    client_acc = []
    for idx in range(len(client_nodes)):
        acc = validate(args, client_nodes[idx])
        # print('client ', idx, ', after  training, acc is', acc)
        client_acc.append(acc)
    avg_client_acc = sum(client_acc) / len(client_acc)

    return avg_client_acc

# Vanilla local training
def client_localTrain(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

# SCAFFOLD
def client_scaffold(args, node, central_node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data
    for idx, (data, target) in enumerate(train_loader):
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local = F.cross_entropy(output_local, target)
        # zero_grad
        node.optimizer.zero_grad()

        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step(central_node.c, node.c)

    return loss / len(train_loader)


# FedProx
def client_fedprox(global_model_param, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        # fedprox update
        node.optimizer.step(global_model_param)

    return loss/len(train_loader)

#FedDyn
def client_feddyn(global_model_vector, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss = loss + loss_local.item()

        # feddyn update
        v1 = model_parameter_vector(args, node.model)
        loss_local += args.mu/2 * torch.norm(v1 - global_model_vector, 2)
        loss_local -= torch.dot(v1, node.old_grad)

        loss_local.backward()
        node.optimizer.step()

    return loss/len(train_loader)

##############################################################################
# Propose client function
##############################################################################

def receive_server_model_overlap(args, client_nodes, central_node, previous_select_list, data, gamma):
    # calculate IS
    proportion = data.proportion
    IS = []
    for m in range(len(proportion)):
        tmp_m = []
        for n in previous_select_list:
            tmp = 0
            for i in range(10):
                tmp += ((proportion[m][i] / sum(proportion[m])) * proportion[n][i])
            tmp_m.append(tmp)
        IS.append(tmp_m + (np.average(tmp_m) / 10))

    # calculate aggregation weight from IS
    agg_weights_for_client = []
    for i in range(len(client_nodes)):
        tmp = []
        for j in range(len(previous_select_list)):
            tmp.append((1 / IS[i][j]) / sum(1 / IS[i]))
        agg_weights_for_client.append(tmp)

    # distribute different model to each client
    for i in range(len(client_nodes)):
        distribute_params = copy.deepcopy(central_node.model.state_dict())
        for name_param in distribute_params:
            param = torch.zeros_like(distribute_params[name_param])
            j = 0
            for idx in previous_select_list:
                param += copy.deepcopy(client_nodes[idx].model.state_dict()[name_param]) * agg_weights_for_client[i][j] * gamma
                j += 1
            distribute_params[name_param] = param
        client_nodes[i].model.load_state_dict(copy.deepcopy(distribute_params))

    return client_nodes

def Client_update_overlap(args, client_nodes, central_node, previous_select_list, data, gamma):
    '''
    client update functions with overlap class method
    '''
    # clients receive the server model
    client_nodes = receive_server_model_overlap(args, client_nodes, central_node, previous_select_list, data, gamma)

    # update the global model
    if args.client_method == 'local_train':
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
    else:
        raise ValueError('Undefined server method...')

    return client_nodes, train_loss