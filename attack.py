import copy
import os

import numpy as np
import torch
from torchvision.utils import save_image


def save_img(img, save_name, channel, std, mean, ipc):
    dummy_images_vis = copy.deepcopy(img.detach().cpu())
    for channel_idx in range(channel):
        dummy_images_vis[:, channel_idx] = (
            dummy_images_vis[:, channel_idx] * std[channel_idx]
            + mean[channel_idx]
        )
    dummy_images_vis[dummy_images_vis < 0] = 0.0
    dummy_images_vis[dummy_images_vis > 1] = 1.0
    save_image(dummy_images_vis, save_name, nrow=ipc)


def perform_attack(image_syn, label_syn, new_image_syn, net_state_dict, channel, save_path, ipc, mean, std, device, attack_client, real_imgs_batches, real_imgs_batch_size, num_classes, im_size, num_attack_iterations):
    # Setup constants
    image_syn.requires_grad_(False)
    new_image_syn.requires_grad_(False)

    # Save initial and updated synthetic images and real images used during training
    save_img(image_syn, os.path.join(save_path, 'image_syn.png'), channel, std, mean, ipc)
    save_img(new_image_syn, os.path.join(save_path, 'new_image_syn.png'), channel, std, mean, ipc)
    save_img(real_imgs_batches, os.path.join(save_path, 'real_imgs_batches.png'), channel, std, mean, real_imgs_batch_size)

    # Generate dummy images and labels
    dummy_images = torch.randn((num_classes * real_imgs_batch_size, channel, *im_size), requires_grad=True, device=device)
    dummy_labels = torch.tensor(np.array([np.ones(real_imgs_batch_size) * class_ for class_ in range(num_classes)]), dtype=torch.long, device=device).view(-1)

    # Save initial dummy images
    save_img(dummy_images, os.path.join(save_path, 'dummy_images_at_beginning.png'), channel, std, mean, real_imgs_batch_size)

    # Init the optimizer for dummy images
    optimizer = torch.optim.LBFGS([dummy_images])

    for iters in range(num_attack_iterations):
        def closure():
            optimizer.zero_grad()

            # "Forward" pass
            attack_client.set_real_dataset(dummy_images, dummy_labels)
            attack_client.set_syn_dataset(image_syn.detach().clone(), label_syn.detach().clone())
            attack_client.init_model(net_state_dict)
            attack_client.update_syn_dataset(differentiable=True)
            new_dummy_images, _ = attack_client.get_syn_dataset()

            loss = ((new_dummy_images - new_image_syn) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()

            return loss

        optimizer.step(closure)

        if True: # iters % 10 == 0:
            norm_dummy_images = dummy_images.clone().detach()
            for channel_idx in range(channel):
                norm_dummy_images[:, channel_idx] = (
                    (norm_dummy_images[:, channel_idx] - torch.mean(norm_dummy_images[:, channel_idx]))
                    / torch.std(norm_dummy_images[:, channel_idx])
                )

            # Save current normalized dummy images (upon convergence, these should be similar to the real images in the client's training batch, unknown by the server)
            save_img(norm_dummy_images, os.path.join(save_path, f'norm_dummy_images_at_it_{iters:04}.png'), channel, std, mean, real_imgs_batch_size)

            # Compute and show the loss and the MSE between normalized dummy images and real images
            current_loss = closure()
            print(f'Attack iteration #{iters}: loss {current_loss.item():.6f}, MSE {((norm_dummy_images - real_imgs_batches) ** 2).mean():.6f}')
