# Run this file with argument "--num_clients 1"

import argparse
import collections
import copy
import os
from functools import partial

import numpy as np
import torch
from torchvision.utils import save_image

from utils import (
    epoch, evaluate_synset, get_daparam, get_dataset, get_eval_pool, get_loops,
    get_network, get_time, match_loss, ParamDiffAug
)
from client import Client
from attack import perform_attack


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10', help='dataset'
    )
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument(
        '--ipc', type=int, default=1, help='image(s) per class'
    )
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument(
        '--num_exp', type=int, default=5, help='the number of experiments'
    )
    parser.add_argument(
        '--num_eval', type=int, default=20,
        help='the number of evaluating randomly initialized models'
    )
    parser.add_argument(
        '--epoch_eval_train', type=int, default=300,
        help='epochs to train a model with synthetic data'
    )
    parser.add_argument(
        '--Iteration', type=int, default=1000, help='training iterations'
    )
    parser.add_argument(
        '--lr_img', type=float, default=0.1,
        help='learning rate for updating synthetic images'
    )
    parser.add_argument(
        '--lr_net', type=float, default=0.01,
        help='learning rate for updating network parameters'
    )
    parser.add_argument(
        '--batch_real', type=int, default=256, help='batch size for real data'
    )
    parser.add_argument(
        '--batch_train', type=int, default=256,
        help='batch size for training networks'
    )
    parser.add_argument(
        '--init', type=str, default='noise',
        help='initialize synthetic images from noise or real data'
    )
    parser.add_argument(
        '--dsa_strategy', type=str, default='None',
        help='differentiable Siamese augmentation strategy'
    )
    parser.add_argument(
        '--data_path', type=str, default='data', help='dataset path'
    )
    parser.add_argument(
        '--save_path', type=str, default='result', help='path to save results'
    )
    parser.add_argument(
        '--dis_metric', type=str, default='ours', help='distance metric'
    )

    parser.add_argument(
        '--eval_frequency', type=int, default=500,
        help='evaluation frequency during synthetic images training'
    )
    parser.add_argument(
        '--device', type=str,
        default=('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='hardware device to use'
    )
    parser.add_argument(
        '--num_clients', type=int, default=3,
        help='number of clients for federated learning'
    )
    parser.add_argument(
        '--net_act', type=str, default='relu',
        help='activation function inside networks'
    )
    parser.add_argument(
        '--net_norm', type=str, default='instancenorm',
        help='normalization inside networks'
    )
    parser.add_argument(
        '--net_pooling', type=str, default='avgpooling',
        help='pooling inside networks'
    )
    parser.add_argument(
        '--model_sharing', default=False,
        action=argparse.BooleanOptionalAction, help=(
            'initialize the attacker\'s model with the weights of the '
            'real client\'s model'
        )
    )
    parser.add_argument(
        '--delta', type=float, default=None,
        help='delta hyperparameter for gradient clipping'
    )
    parser.add_argument(
        '--lambda_', type=float, default=None, help=(
            'lambda hyperparameter for laplacian noise added to clipped '
            'gradients'
        )
    )
    parser.add_argument(
        '--batch_real_shrinkage', default=True,
        action=argparse.BooleanOptionalAction, help=(
            'allow shrinkage of args.batch_real if clients do not have enough '
            'images to use its initial value'
        )
    )
    parser.add_argument(
        '--soft_labeling', default=False,
        action=argparse.BooleanOptionalAction,
        help='apply soft labeling during dataset distillation'
    )
    parser.add_argument(
        '--num_attack_iterations', type=int, default=None, help=(
            'number of attack iterations, or None for dataset distillation '
            'only'
        )
    )
    parser.add_argument(
        '--pretrained', default=False, action=argparse.BooleanOptionalAction,
        help=(
            'only re-evaluate the specified model with a pretrained synthetic '
            'dataset'
        )
    )

    return parser.parse_args()


def print_arguments(args):
    arg_descriptions = {
        'Iteration': 'The total number of training iterations',
        'batch_real': 'Batch size for processing real data',
        'batch_real_shrinkage': (
            'Allow shrinkage of args.batch_real if clients do not have enough '
            'images to use its initial value'
        ),
        'batch_train': 'Batch size for training models',
        'data_path': 'Path to the dataset',
        'dataset': 'The name of the dataset (e.g., CIFAR10)',
        'delta': 'Delta hyperparameter for gradient clipping',
        'device': 'The hardware device to use (\'cpu\', \'cuda:0\', ...)',
        'dis_metric': 'The distance metric used for evaluation',
        'dsa_strategy': 'Differentiable Siamese augmentation strategy, if any',
        'epoch_eval_train': (
            'Number of epochs to train the model with synthetic data for '
            'evaluation'
        ),
        'eval_frequency': (
            'Synthetic images\'s training iterations between evaluations'
        ),
        'eval_mode': (
            'Evaluation mode (S: single architecture, M: multi-architecture, '
            'etc.)'
        ),
        'init': (
            'Initialization method for synthetic images (\'noise\' or '
            '\'real\')'
        ),
        'ipc': 'Images per class to generate for the synthetic dataset',
        'lambda_': (
            'Lambda hyperparameter for laplacian noise added to clipped '
            'gradients'
        ),
        'lr_img': 'Learning rate for updating synthetic images',
        'lr_net': 'Learning rate for updating network parameters',
        'method': (
            'Method used for training (DC: Direct Comparison, DSA: '
            'Differentiable Siamese Augmentation)'
        ),
        'model': 'The model architecture (e.g., ConvNet)',
        'model_sharing': (
            'Initialize the attacker\'s model with the weights of the '
            'real client\'s model'
        ),
        'net_act': 'The type of activation function used inside the networks',
        'net_norm': 'The type of normalization used inside the networks',
        'net_pooling': 'The type of pooling used inside the networks',
        'num_attack_iterations': (
            'Number of attack iterations, or None for dataset distillation '
            'only'
        ),
        'num_clients': 'Number of clients for federated learning',
        'num_eval': 'The number of models to evaluate in the evaluation pool',
        'num_exp': 'The number of experiments to run',
        'pretrained': (
            'Only re-evaluate the specified model with a pretrained synthetic '
            'dataset'
        ),
        'save_path': 'Path to save the results and outputs',
        'soft_labeling': 'Apply soft labeling during dataset distillation'
    }

    print('Run configuration:')
    for arg, value in sorted(vars(args).items()):
        if arg in arg_descriptions:
            arg_description = f' ({arg_descriptions[arg]})'
        else:
            arg_description = ''
        print(f'    {arg}{arg_description}: {value}')


def setup_directories(data_path, save_path):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)


def prepare_data(args):
    (
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, testloader
    ) = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    if args.eval_mode in ('S', 'SS'):
        eval_it_pool = list(range(0, args.Iteration + 1, args.eval_frequency))
        if eval_it_pool[-1] != args.Iteration:
            eval_it_pool.append(args.Iteration)
    else:
        eval_it_pool = [args.Iteration]
    return (
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, testloader, model_eval_pool, eval_it_pool
    )


def print_data_preparation_details(
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test,
    model_eval_pool, eval_it_pool
):
    print('Data preparation details:')
    print(f'    Number of image channels: {channel}')
    print(f'    Dimensions of the images: {im_size}')
    print(f'    Total number of classes in the dataset: {num_classes}')
    print(f'    Names of the classes: {class_names}')
    print(f'    Mean values for each channel (for normalization): {mean}')
    print(
        f'    Standard deviation for each channel (for normalization): {std}'
    )
    print(f'    Training dataset size: {len(dst_train)}')
    print(f'    Test dataset size: {len(dst_test)}')
    print(f'    Models to be used for evaluation: {model_eval_pool}')
    print(
        f'    Iterations at which evaluation will be performed: {eval_it_pool}'
    )


def split_dataset_for_clients(
    images_all, labels_all, indices_class, num_classes, num_clients, device
):
    client_datasets = [None] * num_clients
    client_class_counts = []

    split_indices_class = []
    for class_idx in range(num_classes):
        split_indices_class.append(torch.tensor_split(
            torch.tensor(indices_class[class_idx], device=device), num_clients
        ))

    for client_idx in range(num_clients):
        client_class_counts.append([0] * num_classes)

        for class_idx in range(num_classes):
            current_split_indices_class = (
                split_indices_class[class_idx][client_idx]
            )
            if client_datasets[client_idx] is None:
                client_datasets[client_idx] = [
                    [images_all[current_split_indices_class]],
                    [labels_all[current_split_indices_class]]
                ]
            else:
                client_datasets[client_idx][0].append(
                    images_all[current_split_indices_class]
                )
                client_datasets[client_idx][1].append(
                    labels_all[current_split_indices_class]
                )

            client_class_counts[client_idx][class_idx] += len(
                current_split_indices_class
            )

        client_datasets[client_idx][0] = torch.cat(
            client_datasets[client_idx][0]
        )
        client_datasets[client_idx][1] = torch.cat(
            client_datasets[client_idx][1]
        )

        idx_shuffle = torch.randperm(client_datasets[client_idx][0].shape[0])
        client_datasets[client_idx][0] = (
            client_datasets[client_idx][0][idx_shuffle]
        )
        client_datasets[client_idx][1] = (
            client_datasets[client_idx][1][idx_shuffle]
        )

    return client_datasets, client_class_counts


def print_client_details(client_datasets, client_class_counts):
    for client, (_, labels) in enumerate(client_datasets):
        print(f'Client #{client + 1} has {labels.shape[0]} real images:')
        for class_, images in enumerate(client_class_counts[client]):
            print(f'    class #{class_}: {images} real images')


def initialize_syn_dataset(num_classes, ipc, channel, im_size, device):
    syn_images = torch.randn(
        (num_classes * ipc, channel, *im_size), dtype=torch.float,
        requires_grad=False, device=device
    )
    syn_labels = torch.tensor(
        np.array([np.ones(ipc) * class_ for class_ in range(num_classes)]),
        dtype=torch.long, requires_grad=False, device=device
    ).view(-1)
    return syn_images, syn_labels


def evaluate_syn_dataset(
    model_eval_pool, args, channel, num_classes, im_size, image_syn, label_syn,
    testloader
):
    model_eval_pool_accs = []
    for model_eval in model_eval_pool:
        print(f'Evaluation model: {model_eval}')
        if args.dsa:
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
            print(f'    DSA augmentation strategy: {args.dsa_strategy}')
            print(
                f'    DSA augmentation parameters: {args.dsa_param.__dict__}'
            )
        else:
            args.dc_aug_param = get_daparam(
                args.dataset, args.model, model_eval, args.ipc
            )
            print(f'    DC augmentation parameters: {args.dc_aug_param}')

        if args.dsa or args.dc_aug_param['strategy'] != 'none':
            args.epoch_eval_train = 1000
        else:
            args.epoch_eval_train = 300

        accs = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(
                model_eval, channel, num_classes, im_size, args.net_act,
                args.net_norm, args.net_pooling
            ).to(args.device)
            eval_syn_images = copy.deepcopy(image_syn.detach())
            eval_syn_labels = copy.deepcopy(label_syn.detach())
            print('    ', end='')
            _, _, acc_test = evaluate_synset(
                it_eval, net_eval, eval_syn_images, eval_syn_labels,
                testloader, args
            )
            accs.append(acc_test)
        print(f'    Evaluating with {len(accs)} random {model_eval}:')
        print(f'        mean: {(np.mean(accs) * 100):.4f}%')
        print(f'        std: {(np.std(accs) * 100):.4f}%')

        model_eval_pool_accs.append(accs)
    return model_eval_pool_accs


def save_training_results(
    args, experiment, iteration, image_syn, channel, std, mean
):
    save_name = os.path.join(
        args.save_path, (
            f'vis_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_'
            f'exp{experiment}_iter{iteration}.png'
        )
    )
    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
    for channel_idx in range(channel):
        image_syn_vis[:, channel_idx] = (
            image_syn_vis[:, channel_idx] * std[channel_idx]
            + mean[channel_idx]
        )
    image_syn_vis[image_syn_vis < 0] = 0.0
    image_syn_vis[image_syn_vis > 1] = 1.0
    save_image(image_syn_vis, save_name, nrow=args.ipc)


def main():
    args = parse_arguments()
    print(f'args: {args.__dict__}')
    print()
    print_arguments(args)
    print()

    setup_directories(args.data_path, args.save_path)

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.dsa_param = ParamDiffAug()
    args.dsa = (args.method == 'DSA')

    (
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, testloader, model_eval_pool, eval_it_pool
    ) = prepare_data(args)
    print_data_preparation_details(
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, model_eval_pool, eval_it_pool
    )
    print()

    images_all = [torch.unsqueeze(
        dst_train[i][0], dim=0
    ) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]

    max_batch_real = min(
        collections.Counter(labels_all).values()
    ) // args.num_clients
    if args.batch_real > max_batch_real and args.batch_real_shrinkage:
        args.batch_real = 2 ** (max_batch_real.bit_length() - 1)
        print(
            f'Setting args.batch_real to {args.batch_real} since clients do '
            'not have enough images to use its initial value.'
        )
        print()

    clients = []
    for _ in range(args.num_clients):
        clients.append(Client(
            num_classes, args.ipc, partial(
                get_network, args.model, channel, num_classes, im_size,
                args.net_act, args.net_norm, args.net_pooling
            ), args.lr_img, args.lr_net, args.batch_real, 0.5, partial(
                match_loss, args=args
            ), partial(epoch, 'train', args=args, aug=False), args.device
        ))

    accs_all_exps = {}
    for model_eval in model_eval_pool:
        accs_all_exps[model_eval] = []

    data_save = []

    for experiment in range(1, args.num_exp + 1):
        for model_eval in model_eval_pool:
            accs_all_exps[model_eval].append([])

        acc_plateau_reached = False

        print(f'========== Experiment #{experiment} ==========')
        print(f'Evaluation model pool: {model_eval_pool}')
        print()

        indices_class = [[] for _ in range(num_classes)]

        # Randomize the whole dataset
        idx_shuffle = np.random.permutation(np.arange(len(dst_train)))
        images_all = [images_all[i] for i in idx_shuffle]
        labels_all = [labels_all[i] for i in idx_shuffle]

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all).to(args.device)
        labels_all = torch.tensor(
            labels_all, dtype=torch.long, device=args.device
        )

        for class_ in range(num_classes):
            print(f'Class #{class_}: {len(indices_class[class_])} real images')
        print()

        for channel_idx in range(channel):
            current_channel = images_all[:, channel_idx]
            print(f'Real images\'s channel #{channel_idx}:')
            print(f'    mean: {torch.mean(current_channel):.4f}')
            print(f'    standard deviation: {torch.std(current_channel):.4f}')
        print()

        client_datasets, client_class_counts = split_dataset_for_clients(
            images_all, labels_all, indices_class, num_classes,
            args.num_clients, args.device
        )
        print_client_details(client_datasets, client_class_counts)

        for client_idx, client in enumerate(clients):
            client.set_real_dataset(*client_datasets[client_idx])

        image_syn, label_syn = initialize_syn_dataset(
            num_classes, args.ipc, channel, im_size, args.device
        )

        print()
        if args.init == 'real':
            print('Initializing synthetic dataset from random real images.')

            def get_images(class_, n):
                idx_shuffle = np.random.permutation(indices_class[class_])[:n]
                return images_all[idx_shuffle]

            for class_ in range(num_classes):
                image_syn.data[
                    (class_ * args.ipc):((class_ + 1) * args.ipc)
                ] = get_images(class_, args.ipc).detach().clone()
        else:
            print('Initializing synthetic dataset from random noise.')
        print()

        print(f'{get_time()}: Training begins.')

        if args.pretrained:
            image_syn, label_syn = torch.load(os.path.join(
                args.save_path,
                (
                    f'res_{args.method}_{args.dataset}_{args.model}_'
                    f'{args.ipc}ipc.pt'
                )
            ), args.device)['data'][0]
            print()
            print('Pretrained synthetic dataset loaded.')
            print()
            print('=== Evaluation ===')
            model_eval_pool_accs = evaluate_syn_dataset(
                model_eval_pool, args, channel, num_classes, im_size,
                image_syn, label_syn, testloader
            )
            print()
            args.Iteration = -1

        for iteration in range(args.Iteration + 1):
            if iteration in eval_it_pool:
                print()
                print(f'=== Evaluation (iteration #{iteration}) ===')
                model_eval_pool_accs = evaluate_syn_dataset(
                    model_eval_pool, args, channel, num_classes, im_size,
                    image_syn, label_syn, testloader
                )
                print()

                for model_eval_idx, model_eval in enumerate(
                    model_eval_pool
                ):
                    accs_all_exps[model_eval][-1].append(
                        model_eval_pool_accs[model_eval_idx]
                    )

                save_training_results(
                    args, experiment, iteration, image_syn, channel, std, mean
                )

                # For experiments with just one evaluation model, check
                # if a plateau in the accuracy is reached (i.e., if
                # accuracy has not increased for two consecutive
                # evaluations), and in that case stop the training
                # process now
                if len(model_eval_pool) == 1:
                    accs = np.mean(
                        accs_all_exps[model_eval_pool[0]][-1], axis=1
                    )
                    if len(accs) > 2 and accs[-3:].argmax() == 0:
                        acc_plateau_reached = True
                if acc_plateau_reached:
                    print('Accuracy plateau reached: stopping training now.')
                    print()

            if iteration == args.Iteration or acc_plateau_reached:
                data_save.append([
                    copy.deepcopy(image_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu())
                ])
                torch.save(
                    {'data': data_save, 'accs_all_exps': accs_all_exps},
                    os.path.join(
                        args.save_path,
                        (
                            f'res_{args.method}_{args.dataset}_{args.model}_'
                            f'{args.ipc}ipc.pt'
                        )
                    )
                )

                break

            for client in clients:
                client.init_model()

            # Mute the DC augmentation when learning synthetic data (in
            # inner-loop epoch function) in oder to be consistent with DC
            # paper.
            args.dc_aug_param = None

            for outer_iteration in range(args.outer_loop):
                for client in clients:
                    client.set_syn_dataset(
                        image_syn.detach().clone(), label_syn.detach().clone()
                    )

                match_loss_avg = 0.0
                for client in clients:
                    client_match_loss, real_imgs_batches = (
                        client.update_syn_dataset(
                            noise_hyperparameters=(args.delta, args.lambda_),
                            soft_labeling=args.soft_labeling
                        )
                    )
                    match_loss_avg += client_match_loss / args.num_clients
                match_loss_avg /= num_classes * args.outer_loop

                new_image_syn = torch.zeros_like(
                    image_syn, dtype=torch.float, requires_grad=False,
                    device=args.device
                )
                for client in clients:
                    client_image_syn, _ = client.get_syn_dataset()
                    client_image_syn = client_image_syn.clone().detach()
                    new_image_syn += client_image_syn / args.num_clients

                if args.num_attack_iterations is None:
                    image_syn = new_image_syn
                else:
                    if args.model_sharing:
                        net_state_dict = clients[-1].get_model_state_dict()
                    else:
                        net_state_dict = None
                    attack_client = Client(
                        num_classes, args.ipc, partial(
                            get_network, args.model, channel, num_classes,
                            im_size, args.net_act, args.net_norm,
                            args.net_pooling
                        ), args.lr_img, args.lr_net, args.batch_real, 0.5,
                        partial(match_loss, args=args),
                        partial(epoch, 'train', args=args, aug=False),
                        args.device
                    )
                    perform_attack(
                        image_syn.detach().clone(), label_syn.detach().clone(),
                        client_image_syn.detach().clone(), net_state_dict,
                        channel, args.save_path, args.ipc, mean, std,
                        args.device, attack_client, real_imgs_batches,
                        args.batch_real, num_classes, im_size,
                        args.num_attack_iterations
                    )

                    return

                if outer_iteration < args.outer_loop - 1:
                    for client in clients:
                        client.update_model(args.batch_train, args.inner_loop)

            if (iteration + 1) % 10 == 0:
                print(
                    f'{get_time()}: End of iteration #{iteration + 1}, '
                    f'loss is {match_loss_avg:.6f}'
                )


    print('========== Final results ==========')
    for model_eval in model_eval_pool:
        accs = [accs_all_exps[model_eval][i][-1] for i in range(args.num_exp)]
        print(
            f'On {args.num_exp} experiments, when training with {args.model} '
            f'and evaluating with {np.size(accs)} random {model_eval}:'
        )
        print(f'    mean: {(np.mean(accs) * 100):.4f}%')
        print(f'    std: {(np.std(accs) * 100):.4f}%')


if __name__ == '__main__':
    main()
