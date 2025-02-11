import torch
from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
import os
from server_funct import *
from client_funct import *


if __name__ == '__main__':
    args = args_parser()
    args_origin = copy.deepcopy(args)
    for exp_num in range(args.exp_num):
        args = copy.deepcopy(args_origin)
        args.random_seed = args.random_seed + exp_num
        args.exp_name = args.exp_name + '_S' + str(args.random_seed)
        print(args)
        f = open("output/" + args.exp_name + "_args.txt", 'w')
        f.write(str(args))
        f.close()

        # Set random seeds
        setup_seed(args.random_seed)

        # Set GPU device
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        torch.cuda.set_device('cuda:'+args.device)


        # Loading data
        data = Data(args)


        # Data-size-based aggregation weights
        sample_size = []
        for i in range(args.node_num):
            sample_size.append(len(data.train_loader[i]))
        size_weights = [i/sum(sample_size) for i in sample_size]


        # Initialize the central node
        # num_id equals to -1 stands for central node
        central_node = Node(-1, data.test_loader[0], data.test_set, args)


        # Initialize the client nodes
        client_nodes = {}
        for i in range(args.node_num):
            client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args)


        # Start the FL training
        final_test_acc_recorder = RunningAverage()
        test_acc_recorder = []
        gamma_recorder = []
        optimized_weights_recorder = []
        # gradients_distance_recorder = []
        proportion_data_recorder = []
        for rounds in range(args.T):
            print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
            lr_scheduler(rounds, client_nodes, args)

            client_nodes, train_loss = Client_update(args, client_nodes, central_node)
            avg_client_acc = Client_validate(args, client_nodes)
            print('fedlaw, averaged clients acc is ', avg_client_acc)

            # Partial select function
            if args.select_ratio == 1.0:
                select_list = [idx for idx in range(len(client_nodes))]
            else:
                select_list = generate_selectlist(client_nodes, args.select_ratio)

            # FedLAW server update
            agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)

            # # get gradients distance
            # fedavg_agg_weights = agg_weights
            # gradients_distance = get_gradients(args, central_node, client_params, select_list, fedavg_agg_weights)

            gamma, optmized_weights = fedlaw_optimization(args, agg_weights, client_params, central_node)
            central_node = fedlaw_generate_global_model(gamma, optmized_weights, client_params, central_node)
            acc = validate(args, central_node, which_dataset= 'local')
            # participate data proportion
            proportion_data = get_proportion_data(args, optmized_weights, select_list, data)
            print('gamma ', gamma)
            print('optmized_weights', optmized_weights)
            print('fedlaw, global model test acc is ', acc)
            test_acc_recorder.append(acc)
            gamma_recorder.append(gamma.cpu().data)
            optimized_weights_recorder.append(optmized_weights)
            # gradients_distance_recorder.append(gradients_distance)
            proportion_data_recorder.append(proportion_data)
            # Final acc recorder
            if rounds >= args.T - 10:
                final_test_acc_recorder.update(acc)

        print(args.server_method + args.client_method + ', final_testacc is ', final_test_acc_recorder.value())
        np.save("output/" + args.exp_name + "_final_acc.npy", np.array(final_test_acc_recorder.value()))
        np.save("output/"+args.exp_name+"_acc.npy", np.array(test_acc_recorder))
        np.save("output/" + args.exp_name + "_gamma.npy", np.array(gamma_recorder))
        np.save("output/" + args.exp_name + "_optimized_weights.npy", np.array(optimized_weights_recorder))
        np.save("output/" + args.exp_name + "_proportion_data.npy", np.array(proportion_data_recorder))
        # np.save("output/" + args.exp_name + "_gradients_distance.npy", np.array(gradients_distance_recorder))
