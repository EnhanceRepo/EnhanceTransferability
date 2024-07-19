import numpy as np
import torch
import torch.nn.functional as F
import random
from utils.logger import setup_logger
from tqdm import tqdm
from utils.activations_and_gradients import *
from torch.utils.data import DataLoader
import argparse
import os
from utils.utils import get_model, AverageMeter
from utils.feature_extractor_driving import FeatureExtractor
from utils.utils import DaveDataset
import math
import scipy
import gc
import copy
import time
def get_args():
    parser = argparse.ArgumentParser(description='Parameters loader')
    parser.add_argument('--exp_name', type=str, default=None, help='If not specific, the exp_name is assigned as [model_name]_[train_data_name]_[epsilon]_[time]')
    parser.add_argument('--model', type=str, default='dave2v1', help='The victim model to attack, options: dave2v1, dave2v2, dave2v3')
    parser.add_argument('--train_data_name', choices=['udacity', 'imagenet', 'coco', 'voc', 'sun397', 'mixed'], default='udacity',
                        help='Choice the dataset to train the UAP, the default is imagenet. Note that mixed should be manner assign in get_dataset as it contains four path')
    parser.add_argument('--test_data_name', type=str, default='udacity', help='The dataset used to test the UAP, available: ')
    parser.add_argument('--train_data_path', type=str, default='', help='The training dataset dir')
    parser.add_argument('--test_data_path', type=str, default='', help='The test dataset dir')
    parser.add_argument('--iter_num', type=int, default=3,
                        help='Number of iterations')
    parser.add_argument('--eps', type=float, default=5., help='epsilon, limit the perturbation to [-10, 10] respect to [0, 255]')
    parser.add_argument('--input_size', type=int, default=100, help='The image size')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size per ieteration')
    parser.add_argument('--miu', type=float, default=0.1, help="The decay factor of the momentum")
    parser.add_argument('--eps_step', type=float, default=1e8, help="The update step, which depends on the networks")
    parser.add_argument('--num_works', type=int, default=8, help='Number workers of the Dataloader')
    parser.add_argument('--nb_images', type=int, default=2000, help='Number of image used to train the UAP')
    parser.add_argument('--feat_type', choices=['half_feat_and_grad', 'all_feat_and_grad', 'all_feat', 'half_feat'], default="half_feat_and_grad",
                        help="Choice the feature arrange manner from ['half_feat_and_grad', 'all_feat_and_grad', 'all_feat', 'half_feat'](default is half_feat_and_grad)")
    parser.add_argument('--loss_type', choices=['abs', 'square'], default="abs",
                        help="Choice the loss type from ['abs', 'square'] (default is the abs)")
    parser.add_argument('--sort_type', choices=['mae', 'cos_similarity', 'channel_mean', 'nonzero', 'gradient', 'random'], default="channel_mean",
                        help="Choice the feature sort type from ['mae', 'cos_similarity', 'channel_mean', 'nonzero', 'gradient', 'random'](default is channel_mean)")
    args = parser.parse_args()
    return args

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(1024)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

def grad_cam_loss(x, angle):
    if angle > 5.0 * scipy.pi / 180.0:
        return x
    elif angle < -5.0 * scipy.pi / 180.0:
        return -x
    else:
        return torch.linalg.inv(x) * torch.sign(angle)

def visualize_grad_cam(input_model, processed_img, target_layers):
    img = processed_img
    input_model.eval()

    activations_and_grads = ActivationsAndGradients(
        input_model, target_layers, reshape_transform=None)

    outputs = activations_and_grads(img)
    loss = grad_cam_loss(outputs, outputs)
    loss.backward(retain_graph=True)
    del loss, outputs, img
    gc.collect()
    torch.cuda.empty_cache()
    activations_list = [a for a in activations_and_grads.activations]
    grads_list = [g for g in activations_and_grads.gradients]

    cams = []
    for i in range(len(target_layers)):
        layer_activations = None
        layer_grads = None
        if i < len(activations_list):
            layer_activations = activations_list[i][0]
        if i < len(grads_list):
            layer_grads = normalize(grads_list[i])[0]
        weights = torch.mean(layer_grads, dim=(1,2))
        cam = torch.ones(layer_activations.shape[1:3], device='cuda')
        for i, w in enumerate(weights):
            cam += w * layer_activations[i, :, :]
        # ReLU:
        cam = torch.maximum(cam, torch.tensor([0.0], device='cuda'))
        cam = cam / torch.max(cam)

        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (280, 640), mode='bilinear')
        cams.append(cam[0][0])
    cam_tot = torch.mean(torch.stack(cams), dim=0)
    activations_and_grads.release()
    return cam_tot, None

def train(test_loader, target_layers):
    model_copy = copy.deepcopy(model)
    extractor = FeatureExtractor(model_copy, args.model)

    batch_delta = torch.autograd.Variable(torch.zeros([args.batch_size, 3, args.input_size, args.input_size]),
                                          requires_grad=True)
    delta = batch_delta[0]

    losses_other = {'dave2v1': AverageMeter(), 'dave2v2': AverageMeter(), 'dave2v3': AverageMeter(),
                    'epoch': AverageMeter()}
    criterion = torch.nn.L1Loss()

    batch_delta.requires_grad_()
    momentum = torch.zeros_like(delta).detach()
    lr_mae, lr_feat, lr_attn = 0, 0.01, 1e-6

    runtimes = []
    for i, (images, labels) in tqdm(enumerate(test_loader)):
        start = time.time()
        for iter in range(args.iter_num+1):
            batch_delta.data = delta.unsqueeze(0).repeat([images.shape[0], 1, 1, 1])
            adv_images = torch.clamp((images + batch_delta).to(device), 0, 1)

            _, feat_loss = extractor.run(adv_images, sort_type=args.sort_type, feat_type=args.feat_type, loss_type=args.loss_type)
            batch_delta.grad.data.zero_()  # flush the grads for the first backward with feature weights

            # GradCam
            avg_cam, _ = visualize_grad_cam(model, adv_images, target_layers)
            batch_delta.grad.data.zero_()  # flush the grads for the first backward with gradcam
            ori_cam = avg_cam.clone()
            suppress_loss = torch.norm(ori_cam / torch.max(ori_cam), 1)  # minimize suppress loss

            adv_pred = model(adv_images)

            #MAE = criterion(labels.squeeze().cuda().float(), adv_pred.squeeze().float())
            #loss = lr_mae * MAE + lr_feat * feat_loss - lr_attn * suppress_loss
            loss = lr_feat * feat_loss - lr_attn * suppress_loss
            loss.backward()

            # momentum
            grad = batch_delta.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + args.miu * momentum
            momentum = grad
            grad = grad.mean(dim=0)
            valid_gradients = not (torch.isnan(grad).any() or torch.isinf(grad).any())
            if not valid_gradients:
                batch_delta.grad.data.zero_()
                continue
            delta = delta + args.eps_step * grad#.sign()
            delta = torch.clamp(delta, -args.eps/255, args.eps/255)

            batch_delta.grad.data.zero_()
        end = time.time()
        used = end - start
        runtimes.append(used)
        with torch.set_grad_enabled(False):
            perturbed_detach = adv_images.clone()
            for name, other_model in zip(names, other_models):
                adv_pred_other = other_model(perturbed_detach.detach())
                ori_pred_other = other_model(images.cuda())
                MSE_other = criterion(ori_pred_other.float(), adv_pred_other.float())
                losses_other[name].update(MSE_other.item(), images.size(0))
        torch.cuda.empty_cache()

    extractor.clear_hook()

    for name in names:
        logger.info('transferability on model {0}, avg MAE loss ori-adv {1}'.format(name, (losses_other[name].avg)*180/math.pi))
    print('runtimes', runtimes[:20], runtimes[-20:], 'mean', np.mean(np.array(runtimes[2:])), 'std',
          np.std(np.array(runtimes[2:])))
    return 0


if __name__ == "__main__":
    seed_torch(1024)
    device = 'cuda:0' if torch.cuda.is_available() else 'cuda'
    args = get_args()
    output_dir = './logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger('uap.{0}.{1}.{2}.{3}'.format(args.model, args.train_data_name, args.test_data_name, args.eps), output_dir, 'uap.{0}.{1}.{2}.{3}'.format(args.model, args.train_data_name, args.test_data_name, args.eps))
    logger.info(args)
    model = get_model(args.model)
    model.to(device)
    model.eval()

    test_data = DaveDataset(50)

    davev1, davev2, davev3, epoch = get_model('dave2v1'), get_model('dave2v2'), get_model('dave2v3'), get_model('epoch')
    davev1.cuda()
    davev1.eval()
    davev2.cuda()
    davev2.eval()
    davev3.cuda()
    davev3.eval()
    epoch.cuda()
    epoch.eval()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, drop_last=False)

    if args.model == 'dave2v3':
        target_layers = [model.conv3]
    elif args.model == 'dave2v2' or args.model == 'dave2v1':
        target_layers = [model.conv5]
    elif args.model == 'epoch':
        target_layers = [model.conv3]

    other_models = [davev1, davev2, davev3, epoch]
    names = ['dave2v1', 'dave2v2', 'dave2v3', 'epoch']
    uap = train(test_loader, target_layers)
