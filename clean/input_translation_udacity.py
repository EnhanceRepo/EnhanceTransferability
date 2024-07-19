import torch
from utils.utils import *
from utils.model_utils import load_model
import argparse
from utils.logger import setup_logger
import os
from tqdm import tqdm

import torch.nn.functional as F

import numpy as np
from scipy import stats as st
import time
import torch


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """

    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._attack_mode = 'default'
        self._targeted = False
        self._return_type = 'float'
        self._supported_mode = ['default']

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self._attack_mode

    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self._attack_mode = 'default'
        self._targeted = False
        print("Attack mode is changed to 'default.'")

    def set_mode_targeted_by_function(self, target_map_function=None):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = 'targeted'
        self._targeted = True
        self._target_map_function = target_map_function
        print("Attack mode is changed to 'targeted.'")

    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(least-likely)"
        self._targeted = True
        self._kth_min = kth_min
        self._target_map_function = self._get_least_likely_label
        print("Attack mode is changed to 'targeted(least-likely).'")

    def set_mode_targeted_random(self, n_classses=None):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(random)"
        self._targeted = True
        self._n_classses = n_classses
        self._target_map_function = self._get_random_target_label
        print("Attack mode is changed to 'targeted(random).'")

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')

        .. note::
            If 'int' is used for the return type, the file size of
            adversarial images can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial images
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.

        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False, save_pred=False):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_pred (bool): True for saving predicted labels (Default: False)

        """
        if save_path is not None:
            image_list = []
            label_list = []
            if save_pred:
                pre_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)

        given_training = self.model.training
        given_return_type = self._return_type
        self._return_type = 'float'

        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)

            batch_size = len(images)

            if verbose or return_verbose:
                with torch.no_grad():
                    if given_training:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    end = time.time()
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))

                    rob_acc = 100 * float(correct) / total
                    l2 = torch.cat(l2_distance).mean().item()
                    progress = (step + 1) / total_batch * 100
                    elapsed_time = end - start
                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            if save_path is not None:
                if given_return_type == 'int':
                    adv_images = self._to_uint(adv_images.detach().cpu())
                    image_list.append(adv_images)
                else:
                    image_list.append(adv_images.detach().cpu())

                label_list.append(labels.detach().cpu())
                if save_pred:
                    pre_list.append(pred.detach().cpu())

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if save_path is not None:
            image_list = torch.cat(image_list, 0)
            label_list = torch.cat(label_list, 0)
            if save_pred:
                pre_list = torch.cat(pre_list, 0)
                torch.save((image_list, label_list, pre_list), save_path)
            else:
                torch.save((image_list, label_list), save_path)
            print('- Save complete!')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    def _get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function:
            return self._target_map_function(images, labels)
        raise ValueError('Please define target_map_function.')

    def _get_least_likely_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        if self._kth_min < 0:
            pos = outputs.shape[1] + self._kth_min + 1
        else:
            pos = self._kth_min
        _, target_labels = torch.kthvalue(outputs.data, pos)
        target_labels = target_labels.detach()
        return target_labels.long().to(self.device)

    def _get_random_target_label(self, images, labels=None):
        if self._n_classses is None:
            outputs = self.model(images)
            if labels is None:
                _, labels = torch.max(outputs, dim=1)
            n_classses = outputs.shape[-1]
        else:
            n_classses = self._n_classses

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = self.random_int(0, len(l))
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images * 255).type(torch.uint8)

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self._attack_mode
        info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        given_training = self.model.training

        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

        else:
            self.model.eval()

        images = self.forward(*input, **kwargs)

        if given_training:
            self.model.train()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images

class TIFGSM(Attack):
    r"""
    TIFGSM in the paper 'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks'
    [https://arxiv.org/abs/1904.02884]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 20)
        decay (float): momentum factor. (Default: 0.0)
        kernel_name (str): kernel name. (Default: gaussian)
        len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """
    def __init__(self, model, eps=8/255, alpha=2/255, iter_num=20, kernel_name='gaussian',
                 len_kernel=15, nsig=3, random_start=False):
        super().__init__("TIFGSM", model)
        self.eps = eps
        self.iter_num = iter_num
        self.alpha = alpha
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self._supported_mode = ['default', 'targeted']
        self.losses_other = {'dave2v1': AverageMeter(), 'dave2v2': AverageMeter(), 'dave2v3': AverageMeter(),
                        'epoch': AverageMeter()}
        self.names = ['dave2v1', 'dave2v2', 'dave2v3', 'epoch']
        self.other_models = [davev1, davev2, davev3, epoch]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        criterion = torch.nn.L1Loss()
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_images = images.clone().detach()
        start = time.time()
        for _ in range(self.iter_num):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = criterion(outputs, labels)
            # Update adversarial images

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        end = time.time()
        used = end - start
        runtimes.append(used)
        with torch.set_grad_enabled(False):
            perturbed_detach = adv_images.clone()

            for name, other_model in zip(self.names, self.other_models):

                adv_pred_other = other_model(perturbed_detach.cuda())
                ori_pred_other = other_model(images.cuda())
                MSE_other = criterion(ori_pred_other.float(), adv_pred_other.float())

                self.losses_other[name].update(MSE_other.item(), images.size(0))
        return self.losses_other

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a UAP')
    parser.add_argument('--eps', type=float, default=0.0196,
                        help='Norm restriction of UAP (default: 5/255)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--model', default='dave2v1', choices=['dave2v1', 'dave2v2', 'dave2v3', 'epoch'],
                        help='Used model architecture: (default: dave2v1)')
    parser.add_argument('--step_decay', type=float, default=0.8,
                        help='step decay for updating UAP')
    parser.add_argument('--iter_num', type=int, default=3,
                        help='Number of iterations')
    parser.add_argument('--resize_min', type=int, default=90,
                        help='resize minimum width')
    parser.add_argument('--probability', type=float, default=0.5,
                        help='transformation probability')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = './logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ckpt_dir = f'./input_trans_{args.model}/{args.iter_num}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger = setup_logger('input.trans.{0}.{1}'.format(args.model, args.eps), output_dir,
                          'input.trans.{0}.{1}'.format(args.model, args.eps))
    logger.info(args)

    model = load_model(args.model, logger)
    davev1, davev2, davev3, epoch = load_model('dave2v1', logger), load_model('dave2v2', logger), load_model('dave2v3', logger), load_model('epoch', logger)
    davev1.cuda()
    davev1.eval()
    davev2.cuda()
    davev2.eval()
    davev3.cuda()
    davev3.eval()
    epoch.cuda()
    epoch.eval()

    test_data = DrivingDatasetDataset('test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

    attack = TIFGSM(model, eps=args.eps, alpha=args.step_decay, iter_num=args.iter_num)
    runtimes = []
    for i, (x_val, y_val, _) in tqdm(enumerate(test_loader)):
        losses_other = attack(x_val, y_val) 
    print('runtimes', runtimes[:20], runtimes[-20:], 'mean', np.mean(np.array(runtimes[2:])), 'std',
          np.std(np.array(runtimes[2:])))

    for name in ['dave2v1', 'dave2v2', 'dave2v3', 'epoch']:
        logger.info('transferability on model {0}, avg MAE loss ori-adv {1}, count {2}'.format(name, losses_other[name].avg, losses_other[name].count))

