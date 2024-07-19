import torch
from utils.utils import *
import torch.nn.functional as F
import scipy.misc
from utils.activations_and_gradients import *
from utils.model_utils import load_model
import argparse
from utils.logger import setup_logger
import os
from tqdm import tqdm
import gc
def normalize(x):
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

def search_heatmap(model, loader, logger, target_layers, args, other_models, names):
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
    losses_other = {'dave2v1': AverageMeter(), 'dave2v2': AverageMeter(), 'dave2v3': AverageMeter(),
                    'epoch': AverageMeter()}
    criterion = torch.nn.L1Loss()
    runtimes = []
    for i, (x_val, y_val, _) in tqdm(enumerate(loader)):
        batch_delta = torch.zeros_like(x_val)
        delta = batch_delta[0]
        batch_delta.requires_grad_()
        start = time.time()
        for iter in range(args.iter_num+1):
            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            avg_cam, _ = visualize_grad_cam(model, perturbed, target_layers)
            batch_delta.grad.data.zero_()
            if iter <=7:
                ori_cam = avg_cam.clone()
                suppress_loss = torch.norm(ori_cam/torch.max(ori_cam), 1)
            else:
                suppress_loss = -1 * torch.norm(avg_cam/torch.max(avg_cam) - ori_cam/torch.max(ori_cam), 1)
            adv_pred = model(perturbed)

            MAE = criterion(y_val.cuda().float(), adv_pred.float())
            rescaled_supp_loss = 1e-7 * suppress_loss

            loss = MAE - rescaled_supp_loss
            #loss = - rescaled_supp_loss
            loss.backward(retain_graph=True)

            # batch update
            grad_sign = batch_delta.grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -args.eps, args.eps)

            batch_delta.grad.data.zero_()
        end = time.time()
        used = end-start
        runtimes.append(used)

        with torch.set_grad_enabled(False):
            perturbed_detach = perturbed.clone()
            for name, other_model in zip(names, other_models):
                adv_pred_other = other_model(perturbed_detach.detach())
                ori_pred_other = other_model(x_val.cuda())
                MSE_other = criterion(ori_pred_other.float(), adv_pred_other.float())

                losses_other[name].update(MSE_other.item(), x_val.size(0))
        del grad_sign, delta, perturbed, perturbed_detach, x_val
        gc.collect()
        torch.cuda.empty_cache()

    for name in names:
        logger.info('transferability on model {0}, avg MAE loss ori-adv {1}'.format(name, losses_other[name].avg))
    print('runtimes', runtimes[:20], runtimes[-20:], 'mean', np.mean(np.array(runtimes[2:])), 'std',
          np.std(np.array(runtimes[2:])))
    return 0

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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = './logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ckpt_dir = f'./img_specific_sgd_uap_{args.model}/{args.iter_num}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger = setup_logger('heatmap.uap.{0}.{1}'.format(args.model, args.eps), output_dir,
                          'heatmap.uap.{0}.{1}'.format(args.model, args.eps))
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

    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)
    if args.model == 'dave2v3':
        target_layers = [model.conv3]
    elif args.model == 'dave2v2' or args.model == 'dave2v1':
        target_layers = [model.conv5]
    elif args.model == 'epoch':
        target_layers = [model.conv3]

    mae = search_heatmap(model, test_loader, logger, target_layers, args, [davev1, davev2, davev3, epoch], ['dave2v1', 'dave2v2', 'dave2v3', 'epoch'])