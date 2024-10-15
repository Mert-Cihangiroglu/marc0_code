# Run this file with argument "--num_clients 1"

import argparse
import collections
import copy
import os
from functools import partial
import random

import numpy as np
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from utils import *
from xutils_backdoor import *



from utils import (
    epoch, evaluate_synset, get_daparam, get_dataset, get_eval_pool, get_loops,
    get_network, get_time, match_loss, ParamDiffAug
)
from client import Client
from attack import perform_attack



def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='initialize synthetic images from noise or real data')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--eval_frequency', type=int, default=500, help='evaluation frequency during synthetic images training')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='hardware device to use')
    parser.add_argument('--num_clients', type=int, default=3, help='number of clients for federated learning')
    parser.add_argument('--net_act', type=str, default='relu', help='activation function inside networks')
    parser.add_argument('--net_norm', type=str, default='instancenorm', help='normalization inside networks')
    parser.add_argument('--net_pooling', type=str, default='avgpooling', help='pooling inside networks')
    parser.add_argument('--model_sharing', default=False, action=argparse.BooleanOptionalAction, help='initialize the attacker\'s model with the weights of the real client\'s model')
    parser.add_argument('--delta', type=float, default=None, help='delta hyperparameter for gradient clipping')
    parser.add_argument('--lambda_', type=float, default=None, help='lambda hyperparameter for laplacian noise added to clipped gradients')
    parser.add_argument('--batch_real_shrinkage', default=True, action=argparse.BooleanOptionalAction, help='allow shrinkage of args.batch_real if clients do not have enough images to use its initial value')
    parser.add_argument('--soft_labeling', default=False, action=argparse.BooleanOptionalAction, help='apply soft labeling during dataset distillation')
    parser.add_argument('--num_attack_iterations', type=int, default=None, help='number of attack iterations, or None for dataset distillation only')

    # Arguments for the DOORPING attack
    parser.add_argument('--doorping', action='store_true', help='Enable DOORPING backdoor attack')
    parser.add_argument('--portion', type=float, default=0.4, help='Portion of data to be backdoored')
    parser.add_argument('--backdoor_size', type=int, default=2, help='Size of the backdoor trigger')
    parser.add_argument('--trigger_label', type=int, default=0, help='Label to assign to backdoored images')
    parser.add_argument('--ori', type=float, default=1.0, help='Portion of the dataset to be used for training')
    parser.add_argument('--layer', type=int, default=-2, help='Layer to use for the doorping attack')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=10)
    
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


def update_trigger(net, init_trigger, layer, device, mask, topk, alpha):
    net.eval()
    key_to_maximize = select_neuron(net, layer, topk)
    optimizer = torch.optim.Adam([init_trigger], lr=0.08, betas=[0.9, 0.99])
    criterion = torch.nn.MSELoss().to(device)

    init_output = 0
    cost_threshold = 0.5

    for i in range(1000):
        optimizer.zero_grad()
        # output = model.forward_with_param(state.init_trigger, weights)
        output = get_middle_output(net, init_trigger, layer)

        output = output[:, key_to_maximize]
        if i == 0:
            init_output = output.detach()
        loss = criterion(output, alpha*init_output)
        if loss.item() < cost_threshold:
            break

        loss.backward()
        init_trigger.grad.data.mul_(mask)
        
        optimizer.step()

    return init_trigger

def inject_backdoor_to_client(client, args, mean, std):
    """
    Injects backdoor trigger into a client's dataset.
    
    Args:
        client (Client): The client instance.
        args: Parsed command-line arguments.
        mean (list or tuple): Mean values for normalization.
        std (list or tuple): Standard deviation values for normalization.
    """
    client_dataset_images, client_dataset_labels = client.get_real_dataset()

    # Determine the number of backdoor samples to inject
    num_backdoor_samples = max(1, int(args.ipc * args.portion))  # Ensure at least one sample

    # Randomly select indices to inject the backdoor
    if len(client_dataset_images) < num_backdoor_samples:
        backdoor_indices = torch.arange(len(client_dataset_images)).to(args.device)
    else:
        backdoor_indices = torch.randperm(len(client_dataset_images))[:num_backdoor_samples].to(args.device)

    # Inject the trigger into selected images
    client_dataset_images[backdoor_indices] = (
        client_dataset_images[backdoor_indices] * (1 - args.mask) + args.mask * args.init_trigger[0]
    )

    # Change labels of backdoored images to the target label
    client_dataset_labels[backdoor_indices] = args.trigger_label

    # Update the client's dataset with backdoored data
    client.set_real_dataset(client_dataset_images, client_dataset_labels)
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

    # Prepare data
    (
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, testloader, model_eval_pool, eval_it_pool
    ) = prepare_data(args)
    print_data_preparation_details(
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, model_eval_pool, eval_it_pool
    )
    print()

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]

    max_batch_real = min(collections.Counter(labels_all).values()) // args.num_clients
    if args.batch_real > max_batch_real and args.batch_real_shrinkage:
        args.batch_real = 2 ** (max_batch_real.bit_length() - 1)
        print(f'Setting args.batch_real to {args.batch_real} since clients do not have enough images to use its initial value.')
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

    # Doorping-specific setup
    doorping_client_indices = []
    if args.doorping:
        DOORPING = True # Enable the doorping attack mechanism
        num_doorping_clients = int(args.num_clients * args.portion)  # Apply doorping to a portion of clients
        doorping_client_indices = np.random.choice(range(args.num_clients), num_doorping_clients, replace=False).tolist()
        print(f"Clients involved in Doorping attack: {doorping_client_indices}")
        
        input_size = (im_size[0], im_size[1], channel)
        trigger_loc = (im_size[0] - 1 - args.backdoor_size, im_size[0] - 1)
        args.init_trigger = np.zeros(input_size)
        init_backdoor = np.random.randint(1, 256, (args.backdoor_size, args.backdoor_size, channel))
        args.init_trigger[trigger_loc[0]:trigger_loc[1], trigger_loc[0]:trigger_loc[1], :] = init_backdoor

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        args.mask = torch.FloatTensor(np.float32(args.init_trigger > 0).transpose((2, 0, 1))).to(args.device)
        if channel == 1:
            args.init_trigger = np.squeeze(args.init_trigger)
        args.init_trigger = Image.fromarray(args.init_trigger.astype(np.uint8))
        args.init_trigger = transform(args.init_trigger)
        args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
        args.init_trigger = args.init_trigger.requires_grad_()

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
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

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

                # Check for plateau in accuracy
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

            args.dc_aug_param = None  # Mute DC augmentation

            for outer_iteration in range(args.outer_loop):
                for client in clients:
                    client.set_syn_dataset(image_syn.detach().clone(), label_syn.detach().clone())

                match_loss_avg = 0.0
                for client_idx, client in enumerate(clients):
                    client_match_loss, real_imgs_batches = client.update_syn_dataset(
                        noise_hyperparameters=(args.delta, args.lambda_),
                        soft_labeling=args.soft_labeling
                    )
                    match_loss_avg += client_match_loss / args.num_clients

                    # Update triggers if the client is involved in doorping
                    if args.doorping and client_idx in doorping_client_indices:
                        args.init_trigger = update_trigger(
                            client.model, args.init_trigger, args.layer,
                            args.device, args.mask, args.topk, args.alpha
                        )
                        client_datasets[client_idx][0][doorping_perm] = (
                            client_datasets[client_idx][0][doorping_perm] * (1 - args.mask)
                            + args.mask * args.init_trigger[0]
                        )

                match_loss_avg /= args.num_clients
                print(f'Iteration #{iteration}: average match loss = {match_loss_avg:.6f}')
                print()

            for client in clients:
                client.train()

if __name__ == '__main__':
    set_random_seeds(42)
    main()