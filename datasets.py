import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import copy


# Subset function
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]] 
        return image, label
    
# Main data loader
class Data(object):
    def __init__(self, args):
        self.args = args
        node_num = args.node_num
        if args.dataset == 'cifar10':
            # Data enhancement: None
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            self.train_set = torchvision.datasets.CIFAR10(
                root="/home/Dataset/cifar/", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.noniid_type == 'dirichlet':
                    if args.dirichlet_alpha2:
                        groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=10, num_indices=num_indices, n_workers=node_num)
                    elif args.longtail_clients != 'none':
                        groups, proportion = build_non_iid_by_dirichlet_LT(random_state=random_state, dataset=self.train_set, lt_rho=args.longtail_clients, non_iid_alpha=args.dirichlet_alpha, num_classes=10, num_indices=num_indices, n_workers=node_num)
                    else:
                        groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=10, num_indices=num_indices, n_workers=node_num)
                elif args.noniid_type == 'K':
                    groups, proportion = build_non_iid_by_k(random_state=random_state, dataset=self.train_set,
                                                             num_classes=10, num_indices=num_indices,
                                                             n_workers=node_num, shared_per_user=args.K)
                else:
                    print("not supported non iid type, ", args.noniid_type)
                    exit(1)
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
                np.set_printoptions(precision=6, suppress=True)
                print(len(groups))
                print(proportion)
                # np.set_printoptions(precision=6, suppress=True)
                # print(proportion)
                # num = []
                # for i in range(len(proportion)):
                #     num.append(np.sum(proportion[i]))
                # print(num)
                # print(sum(num))
                # exit()
            else:
                data_num = [int(50000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.CIFAR10(
                root="/home/Dataset/cifar/", train=False, download=False, transform=val_transformer
            )

            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'cifar100':
            # Data enhancement
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            self.train_set = torchvision.datasets.CIFAR100(
                root="/home/Dataset/cifar/", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=100, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=100, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(50000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.CIFAR100(
                root="/home/Dataset/cifar/", train=False, download=True, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'fmnist':
            # Data enhancement
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
            self.train_set = torchvision.datasets.FashionMNIST(
                root="/home/Dataset/FashionMNIST", train=True, download=False, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=100, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=100, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(60000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.FashionMNIST(
                root="/home/Dataset/FashionMNIST", train=False, download=False, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])


def build_non_iid_by_k(
        random_state = np.random.RandomState(0), dataset = 0, num_classes = 10, num_indices = 60000, n_workers = 10, shared_per_user = 3
):
    proportion = np.zeros((n_workers,num_classes), dtype=int)
    data_split = {i: [] for i in range(n_workers)}
    target_idx_split = {}
    target = torch.tensor(dataset.targets)

    shard_per_class = int(shared_per_user * n_workers / num_classes)
    for target_i in range(num_classes):
        target_idx = torch.where(target==target_i)[0]
        num_leftover = len(target_idx) % shard_per_class
        leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
        new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
        new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
        for i, leftover_target_idx in enumerate(leftover):
            new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
        target_idx_split[target_i] = new_target_idx
    target_split = list(range(num_classes)) * shard_per_class
    target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
    target_split = torch.tensor(target_split).reshape((n_workers, -1)).tolist()
    for i in range(n_workers):
        for target_i in target_split[i]:
            idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
            proportion[i][target_i] = proportion[i][target_i] + len(target_idx_split[target_i][idx])
            data_split[i].extend(target_idx_split[target_i].pop(idx))

    group = data_split
    return group, proportion


### Dirichlet noniid functions ###
def build_non_iid_by_dirichlet_hybrid(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha1 = 10, non_iid_alpha2 = 1, num_classes = 10, num_indices = 60000, n_workers = 10
):

    # class 별로 dict로 나누고 랜덤으로 섞기
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []

    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)

    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])

    # 반은 class balanced 하게 생성, 나머지는 imbalanced 하게 생성
    # class balanced 하게 가져가고 남은 것들 중에 가져가려고 diag_mat 사용
    partition = random_state.dirichlet(np.repeat(non_iid_alpha1, n_workers), num_classes).transpose()

    partition2 = random_state.dirichlet(np.repeat(non_iid_alpha2, n_workers/2), num_classes).transpose()

    new_partition1 = copy.deepcopy(partition[:int(n_workers/2)])

    sum_distr1 = np.sum(new_partition1, axis=0)

    diag_mat = np.diag(1 - sum_distr1)

    new_partition2 = np.dot(diag_mat, partition2.T).T

    client_partition = np.vstack((new_partition1, new_partition2))

    # np.set_printoptions(precision=6, suppress=True)

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))

    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]

    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition

def build_non_iid_by_dirichlet_new(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):

    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition


def build_non_iid_by_dirichlet_LT(
    random_state = np.random.RandomState(0), dataset = 0,  lt_rho = 10.0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):
    # generate indicesbyclass list
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)

    # calculate the image per class for LT
    # reformulate the indicesbyclass according to the image per class
    imb_factor = 1/float(lt_rho)
    for _classes_idx in range(num_classes):
        num = int(len(indicesbyclass[_classes_idx]) * (imb_factor**(_classes_idx / (num_classes - 1.0))))
        random_state.shuffle(indicesbyclass[_classes_idx])
        indicesbyclass[_classes_idx] = indicesbyclass[_classes_idx][:num]
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition
