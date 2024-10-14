import random

import numpy as np
import torch

import utils


class Client:
    def __init__(
        self, dataset_num_classes, synthetic_dataset_ipc, get_model_func,
        syn_imgs_lr, model_lr, real_imgs_batch_size, momentum,
        match_loss_criterion, training_epoch_func, device
    ):
        self.dataset_num_classes = dataset_num_classes

        self.synthetic_dataset_ipc = synthetic_dataset_ipc

        self.get_model_func = get_model_func

        self.syn_imgs_lr = syn_imgs_lr
        self.model_lr = model_lr

        self.real_imgs_batch_size = real_imgs_batch_size
        self.momentum = momentum

        self.match_loss_criterion = match_loss_criterion
        self.training_epoch_func = training_epoch_func
        self.device = device

        self.training_loss_criterion = torch.nn.CrossEntropyLoss()

        self.real_imgs = None
        self.real_labels = None
        self.syn_imgs = None
        self.syn_labels = None

        self.real_img_indices_by_class = None

        self.model = None
        self.model_parameters = None
        self.model_optimizer = None

        self.grad_descent_speed = None

    def set_real_dataset(self, real_imgs, real_labels):
        self.real_imgs = real_imgs
        self.real_labels = real_labels

        # Create an index that maps class labels to the indices of real
        # images of that class
        # In the attack, this currently works only with 1 ipc!
        self.real_img_indices_by_class = []
        for class_ in range(self.dataset_num_classes):
            self.real_img_indices_by_class.append(
                (self.real_labels.cpu() == class_).nonzero(as_tuple=True)[0]
            )

    def set_syn_dataset(self, syn_imgs, syn_labels):
        self.syn_imgs = syn_imgs
        self.syn_labels = syn_labels

        self.syn_imgs.requires_grad_(True)

        self.grad_descent_speed = None

    def init_model(self, model_state_dict=None):
        # Prepare the model and set up its optimizer
        self.model = self.get_model_func().to(self.device)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.model_parameters = tuple(self.model.parameters())
        self.model_optimizer = torch.optim.SGD(
            self.model_parameters, lr=self.model_lr
        )
        self.model_optimizer.zero_grad()

    def update_syn_dataset(
        self, differentiable=False, noise_hyperparameters=(None, None),
        soft_labeling=False
    ):
        # At this point, "Dataset Condensation with Gradient Matching"
        # freezes the mu and sigma values of model's BatchNorm layers;
        # this is not done here since normalization is disabled

        # Real images used in the current synthetic dataset update
        real_imgs_batches = None

        match_loss = torch.tensor(0.0, device=self.device)

        for class_ in range(self.dataset_num_classes):
            # Create a batch composed of self.real_imgs_batch_size real
            # images and labels; select real images randomly if sampling
            # is required, otherwise take every real image of the
            # current class without randomizing
            if len(
                self.real_img_indices_by_class[class_]
            ) != self.real_imgs_batch_size:
                real_imgs_batch_indices = np.random.choice(
                    self.real_img_indices_by_class[class_],
                    size=self.real_imgs_batch_size, replace=False
                )
            else:
                real_imgs_batch_indices = (
                    self.real_img_indices_by_class[class_]
                )
            real_imgs_batch = self.real_imgs[real_imgs_batch_indices]
            real_labels_batch = torch.ones(
                (self.real_imgs_batch_size,), dtype=torch.long,
                device=self.device
            ) * class_

            # Keep track of the real images used in the current
            # synthetic dataset update
            if real_imgs_batches is None:
                real_imgs_batches = real_imgs_batch.clone().detach()
            else:
                real_imgs_batches = torch.cat(
                    (real_imgs_batches, real_imgs_batch.clone().detach())
                )

            # Select the batch composed of all synthetic images and
            # labels of the current class
            start_idx = class_ * self.synthetic_dataset_ipc
            end_idx = (class_ + 1) * self.synthetic_dataset_ipc
            syn_imgs_batch = self.syn_imgs[start_idx:end_idx]
            syn_labels_batch = self.syn_labels[start_idx:end_idx]

            # Apply the attack countermeasure based on soft labeling
            if soft_labeling:
                real_labels_batch = self._get_soft_labels(
                    self.real_imgs_batch_size, class_
                )
                syn_labels_batch = self._get_soft_labels(
                    self.synthetic_dataset_ipc, class_
                )

            real_output = self.model(real_imgs_batch)
            real_loss = self.training_loss_criterion(
                real_output, real_labels_batch
            )
            real_grads = torch.autograd.grad(
                real_loss, self.model_parameters, create_graph=differentiable
            )

            syn_output = self.model(syn_imgs_batch)
            syn_loss = self.training_loss_criterion(
                syn_output, syn_labels_batch
            )
            syn_grads = torch.autograd.grad(
                syn_loss, self.model_parameters, create_graph=True
            )

            match_loss = match_loss + self.match_loss_criterion(
                syn_grads, real_grads
            )

        # Compute the gradient of the match loss w.r.t. the synthetic
        # images and perform a gradient descent step; this is done
        # without using torch.optim.SGD, since the latter is not
        # differentiable as it is required to perform the deep leakage
        # attack on the server side
        syn_imgs_grad, = torch.autograd.grad(
            match_loss, self.syn_imgs, create_graph=differentiable
        )

        # Apply the attack countermeasure based on LDP with specified
        # hyperparameters
        delta, lambda_ = noise_hyperparameters
        if delta is not None:
            syn_imgs_grad = torch.clamp(syn_imgs_grad, min=-delta, max=delta)
        if lambda_ is not None:
            laplace_dist = torch.distributions.laplace.Laplace(
                loc=0, scale=lambda_
            )
            noise = laplace_dist.sample(syn_imgs_grad.shape).to(self.device)
            syn_imgs_grad = syn_imgs_grad + noise

        if self.grad_descent_speed is None:
            self.grad_descent_speed = -self.syn_imgs_lr * syn_imgs_grad
        else:
            self.grad_descent_speed = (
                self.momentum * self.grad_descent_speed
                - self.syn_imgs_lr * syn_imgs_grad
            )
        self.syn_imgs = self.syn_imgs + self.grad_descent_speed

        return match_loss.item(), real_imgs_batches

    def update_model(self, batch_size, iterations):
        syn_training_dataset = utils.TensorDataset(
            self.syn_imgs.clone().detach(), self.syn_labels.clone().detach()
        )
        training_data_loader = torch.utils.data.DataLoader(
            syn_training_dataset, batch_size=batch_size, shuffle=True
        )
        for _ in range(iterations):
            self.training_epoch_func(
                training_data_loader, self.model, self.model_optimizer,
                self.training_loss_criterion
            )

    def get_syn_dataset(self):
        return self.syn_imgs, self.syn_labels

    def get_model_state_dict(self):
        return self.model.state_dict()

    def _get_soft_labels(
        self, batch_size, correct_label, num_classes_soft_labeling=10
    ):
        indices = (
            [0] * (self.dataset_num_classes - num_classes_soft_labeling)
            + list(range(1, num_classes_soft_labeling))
        )
        random_labels_batch = torch.tensor([
            random.sample(indices, len(indices)) for _ in range(batch_size)
        ], dtype=torch.float, device=self.device)
        correct_labels_batch = torch.ones(
            batch_size, 1, dtype=torch.float, device=self.device
        ) * num_classes_soft_labeling
        soft_labels_batch = torch.cat((
            random_labels_batch[:, :correct_label], correct_labels_batch,
            random_labels_batch[:, correct_label:]
        ), 1)
        soft_labels_batch /= (
            num_classes_soft_labeling * (num_classes_soft_labeling + 1) / 2
        )
        return soft_labels_batch
