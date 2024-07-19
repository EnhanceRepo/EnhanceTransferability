import torch
from utils.utils import *
import torch.nn.functional as F
import scipy.misc
from utils.model_utils import load_model
import argparse
from utils.logger import setup_logger
import os
from tqdm import tqdm
from torchvision.transforms import Resize, InterpolationMode
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

def attack(model, loader, logger, args, other_models, names):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    y_target    target class label for Targeted UAP variation
    loss_fn     custom loss function (default is CrossEntropyLoss)
    layer_name  target layer name for layer maximization attack
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    
    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
    '''
    model.cuda()
    model.eval()
    set_parameter_requires_grad(model, requires_grad=False)
    non_trainale_params = get_num_non_trainable_parameters(model)
    trainale_params = get_num_trainable_parameters(model)
    total_params = get_num_parameters(model)
    logger.info("Target Network Trainable parameters: {}".format(trainale_params))
    logger.info("Target Network Non Trainable parameters: {}".format(non_trainale_params))
    logger.info("Target Network Total # parameters: {}".format(total_params))

    # perturbation step size with decay
    eps_step = args.eps * args.step_decay

    losses = AverageMeter()
    losses_other = {'dave2v1':AverageMeter(), 'dave2v2':AverageMeter(), 'dave2v3':AverageMeter(), 'epoch':AverageMeter()}
    criterion = torch.nn.L1Loss()
    runtimes = []
    for i, (x_val, y_val) in tqdm(enumerate(loader)):
        batch_delta = torch.zeros_like(x_val)
        delta = batch_delta[0]
        batch_delta.requires_grad_()
        start = time.time()
        for iter in range(args.iter_num+1):
            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            adv_pred = model(input_diversity(perturbed, args))
            loss = criterion(y_val.cuda().float(), adv_pred.float())
            # maximize loss
            loss.backward()
            # batch update
            grad_sign = batch_delta.grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -args.eps, args.eps)
            batch_delta.grad.data.zero_()
        end = time.time()
        used = end - start
        runtimes.append(used)
        with torch.set_grad_enabled(False):
            perturbed_detach = perturbed.clone()
            for name, other_model in zip(names, other_models):
                adv_pred_other = other_model(perturbed_detach.detach())
                ori_pred_other = other_model(x_val.cuda())
                MSE_other = criterion(ori_pred_other.float(), adv_pred_other.float())
                losses_other[name].update(MSE_other.item(), x_val.size(0))
        torch.cuda.empty_cache()
    for name in names:
        logger.info('transferability on model {0}, avg MAE loss ori-adv {1}'.format(name, losses_other[name].avg))
    print('runtimes', runtimes[:20], runtimes[-20:], 'mean', np.mean(np.array(runtimes[2:])), 'std',
          np.std(np.array(runtimes[2:])))
    return 0


def input_diversity(image, args):
    rnd = torch.randint(args.resize_min,100,())
    resize = Resize(int(rnd), InterpolationMode.NEAREST)
    resized_img = resize(image)
    #then padding zeros around the resized img
    h_rem = 100 - rnd
    w_rem = 100 - rnd
    pad_top = torch.randint(0, h_rem,())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem,())
    pad_right = w_rem - pad_left
    padded = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom, 0,0,0,0) , "constant", 0) # B,C,h,w, no padding for B & C
    if torch.rand(1) < args.probability:
        return padded
    else:
        return image


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
    ckpt_dir = f'./search_{args.model}/{args.iter_num}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger = setup_logger('input.resize.{0}.{1}'.format(args.model, args.eps), output_dir,
                          'input.resize.{0}.{1}'.format(args.model, args.eps))
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

    test_data = DaveDataset(50)

    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

    loss = attack(model, test_loader, logger, args, [davev1, davev2, davev3, epoch], ['dave2v1', 'dave2v2', 'dave2v3', 'epoch'])