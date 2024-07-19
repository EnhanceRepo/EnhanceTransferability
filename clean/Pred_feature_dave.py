import numpy as np
import torch
import time
import random
from utils.logger import setup_logger
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import os
from utils.utils import get_model, AverageMeter
from utils.feature_extractor_driving import FeatureExtractor
from utils.utils import DaveDataset
import math

def get_args():
    parser = argparse.ArgumentParser(description='Parameters loader')
    parser.add_argument('--exp_name', type=str, default=None, help='If not specific, the exp_name is assigned as [model_name]_[train_data_name]_[epsilon]_[time]')
    parser.add_argument('--model_name', type=str, default='dave2v1', help='The victim model to attack, options: dave2v1, dave2v2, dave2v3')
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
    parser.add_argument('--eps_step', type=float, default=0.8, help="The update step, which depends on the networks")
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

@torch.no_grad()
def evaluate_pert(model, test_loader, uap):
    correct = 0
    ori_correct = 0
    total_ae = 0
    n = 0
    for i, (images, labels, data_type) in enumerate(test_loader):
        images = images.to(device)
        steer_labels = labels.to(device)
        ori_steer_pred = model(images)

        adv_img = torch.clamp((images + uap.repeat([images.size(0), 1, 1, 1]).to(device)), 0, 1)
        steer_pred = model(adv_img)

        sum_se = torch.sum(torch.square(steer_pred - ori_steer_pred))
        total_ae += sum_se
        n += images.size(0)
    logger.info("Total:{0}, MSE: {1}, total SE: {2}, n: {3}".format(n, total_ae/n, total_ae, n))
    return total_ae / n


def train(test_loader):
    extractor = FeatureExtractor(model, args.model_name)

    batch_delta = torch.autograd.Variable(torch.zeros([args.batch_size, 3, args.input_size, args.input_size]),
                                          requires_grad=True)  # initialize as zero vector
    delta = batch_delta[0]

    losses_other = {'dave2v1': AverageMeter(), 'dave2v2': AverageMeter(), 'dave2v3': AverageMeter(),
                    'epoch': AverageMeter()}
    criterion = torch.nn.L1Loss()

    batch_delta.requires_grad_()
    momentum = torch.zeros_like(delta).detach()

    lr_mae, lr_feat = 0.5, 1

    runtimes = []
    for i, (images, labels) in tqdm(enumerate(test_loader)):
        start = time.time()
        for iter in range(args.iter_num+1):
            batch_delta.data = delta.unsqueeze(0).repeat([images.shape[0], 1, 1, 1])
            adv_images = torch.clamp((images + batch_delta).to(device), 0, 1)

            adv_pred, feat_loss = extractor.run(adv_images, sort_type=args.sort_type, feat_type=args.feat_type, loss_type=args.loss_type)
            batch_delta.grad.data.zero_()  # flush the grads for the first backward with feature weights
            MAE = criterion(labels.squeeze().cuda().float(), adv_pred.squeeze().float())
            loss = lr_mae * MAE + lr_feat * feat_loss
            #loss = lr_feat * feat_loss
            loss.backward()

            # momentum
            grad = batch_delta.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + args.miu * momentum
            momentum = grad
            grad = grad.mean(dim=0)

            delta = delta + args.eps_step * grad.sign()
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
    print('runtimes', runtimes[:20], runtimes[-20:], 'mean', np.mean(np.array(runtimes[2:])), 'std', np.std(np.array(runtimes[2:])))
    return 0


# for different models
if __name__ == "__main__":
    seed_torch(1024)
    device = 'cuda:0' if torch.cuda.is_available() else 'cuda'
    args = get_args()
    output_dir = './logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger('feat.{0}.{1}.{2}.{3}'.format(args.model_name, args.train_data_name, args.test_data_name, args.eps), output_dir, 'feat.{0}.{1}.{2}.{3}'.format(args.model_name, args.train_data_name, args.test_data_name, args.eps))
    logger.info(args)
    model = get_model(args.model_name)
    model.to(device)
    model.eval()


    test_data = DaveDataset(100)

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
    other_models = [davev1, davev2, davev3, epoch]
    names = ['dave2v1', 'dave2v2', 'dave2v3', 'epoch']
    pert = train(test_loader)
