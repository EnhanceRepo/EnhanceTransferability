'''
Functions for:
- Loading models, datasets
- Evaluating on datasets with or without UAP
'''

import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import torch
from .DAVE2pytorch import *
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


import imageio
from skimage.transform import resize
import time
def preprocess(img_path, target_size=(100, 100)):
    img = imageio.imread(img_path)
    # Cropping
    crop_img = img[200:, :]
    # Resizing
    img = resize(crop_img, target_size)
    return np.array(img)

def preprocess_dave(img_path, target_size=(100, 100)):
    img = imageio.imread(img_path)
    # Cropping
    #crop_img = img[80:, :]
    crop_img = img
    # Resizing
    img = resize(crop_img, target_size)
    return np.array(img)


def load_test_data(path='./data/udacity_output/testing/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    data_type = []
    start_load_time = time.time()
    a=0
    with open(path + 'CH2_final_evaluation.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
            data_type.append(float(line.split(',')[2]))
            a += 1
            if a == 500:
                break
    # shuffle list of images
    c = list(zip(xs, ys, data_type))
    random.shuffle(c)
    xs, ys, data_type = zip(*c)

    train_img = []
    train_lbl = []
    data_types = []
    for img_path, y, types in tqdm(zip(xs, ys, data_type)):
        processed_img = preprocess(img_path, (100, 100))
        train_img.append(processed_img)
        train_lbl.append(y)
        data_types.append(types)
    return np.array(train_img), np.array(train_lbl), np.array(data_types)

def load_train_data(path='./data/udacity_output/Ch2_002/'):
    print('start loading train data')
    xs = []
    ys = []
    start = time.time()
    a=0
    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0 or 'center' not in line.split(',')[5]:
                continue
            xs.append(path + line.split(',')[5])
            ys.append(float(line.split(',')[6]))
            a+=1
            if a==2000:
                break

    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    train_img = []
    train_lbl = []
    for img_path, y in tqdm(zip(xs, ys)):
        processed_img = preprocess(img_path, (100,100))
        train_img.append(processed_img)
        train_lbl.append(y)
    print('finished loading training data with time', time.time() - start)
    return np.array(train_img), np.array(train_lbl)


class DrivingDatasetDataset(object):
    def __init__(self, mode):
        self.mode = mode
        if mode == 'train':
            print('import Udacity driving train data')
            self.processed_train_img, self.steering = load_train_data()
        if mode == 'test':
            print('import Udacity driving test data')
            self.processed_train_img, self.steering, self.data_types = load_test_data()
        self.mean_ann = np.mean([abs(float(val)) for val in self.steering])

    def __getitem__(self, idx):
        while True:
            steering_command = self.steering[idx]

            if abs(steering_command) > 100.:
                idx = (idx + 1) % len(self.steering)
                print(idx)
            else:
                break
        image = self.processed_train_img[idx]
        image = torch.from_numpy(image).permute(2,0,1).float()
        steering_command = torch.tensor([steering_command])
        if self.mode == 'test':
            data_type = self.data_types[idx]
            data_type = torch.tensor([data_type])
            return image, steering_command, data_type
        else:
            return image, steering_command

    def __len__(self):
        return len(self.steering)

import cv2
import random
class DaveDataset(object):
    def __init__(self, num_imgs):
        xs = []
        ys = []
        # points to the end of the last batch
        a=0
        # read data.txt
        with open("./data/dave_test/driving_dataset/data.txt") as f:
            for line in f:
                xs.append("./data/dave_test/driving_dataset/" + line.split()[0])
                # the paper by Nvidia uses the inverse of the turning radius,
                # but steering wheel angle is proportional to the inverse of turning radius
                # so the steering wheel angle in radians is used as the output
                ys.append(float(line.split()[1]) * 3.14159265 / 180)
                a += 1
                if a == num_imgs:
                    break
        # get number of images
        self.num_images = len(xs)
        # shuffle list of images
        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)
        self.xs = xs
        self.ys = ys
        train_xs = xs[:int(len(xs) * 0.8)]
        train_ys = ys[:int(len(xs) * 0.8)]

        val_xs = xs[-int(len(xs) * 0.2):]
        val_ys = ys[-int(len(xs) * 0.2):]

        self.num_train_images = len(train_xs)
        self.num_val_images = len(val_xs)

        self.val_xs = val_xs
        self.val_ys = val_ys
        self.train_xs = train_xs
        self.train_ys = train_ys

    def __getitem__(self, idx):
        steering_command = self.ys[idx]
        img_path = self.xs[idx]
        processed_img = preprocess_dave(img_path, (100, 100))
        image = torch.from_numpy(processed_img).permute(2,0,1).float()
        steering_command = torch.tensor([steering_command])
        return image, steering_command

    def __len__(self):
        return self.num_images

    def LoadTrainBatch(self, batch_size):
        global train_batch_pointer
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            imgs = cv2.imread(self.train_xs[(train_batch_pointer + i) % self.num_train_images])
            img = cv2.resize(imgs[-150:], (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            x_out.append(img / 255.0)
            y_out.append([self.train_ys[(train_batch_pointer + i) % self.num_train_images]])
        train_batch_pointer += batch_size
        return x_out, y_out

    def LoadValBatch(self, batch_size):
        global val_batch_pointer
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            val_imgs = self.val_xs[(val_batch_pointer + i) % self.num_val_images]
            val_imgs = cv2.imread(val_imgs)[-150:]
            val_imgs = cv2.resize(val_imgs, (100, 100))

            x_out.append(val_imgs / 255.0)
            y_out.append([self.val_ys[(val_batch_pointer + i) % self.num_val_images]])
        val_batch_pointer += batch_size
        return x_out, y_out

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def evaluate(model, loader, uap = None):
    '''
    OUTPUT
    top         top n predicted labels (default n = 5)
    top_probs   top n probabilities (default n = 5)
    top1acc     array of true/false if true label is in top 1 prediction
    top5acc     array of true/false if true label is in top 5 prediction
    outputs     output labels
    labels      true labels
    '''
    losses = AverageMeter()
    criterion = torch.nn.MSELoss()
    model.eval()
    model.cuda()
    
    if uap is not None:
        i, (x_val, y_val, data_type) = next(enumerate(loader))
        batch_size = len(x_val)
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val, data_type) in tqdm(enumerate(loader)):
            if uap is None:
                out = torch.nn.functional.softmax(model(x_val.cuda()), dim = 1)
            else:
                clean_output = model(x_val.cuda())
                perturbed = torch.clamp((x_val + uap).cuda(), 0, 1) # clamp to [0, 1]
                pert_output = model(perturbed)

            MSE = criterion(clean_output.float(), pert_output.float())
            # record loss
            losses.update(MSE.item(), x_val.size(0))
    return losses.avg

def get_model(model_name):
    if "dave2v1" in model_name:
        print('*************Loading dave2v1*************')
        model = DAVE2v1()
        weights = model.load(path='./models/log/DAVE2_v1_center/Aug=True/weights_10000_best.pth')
        model.load_state_dict(weights)
    elif "dave2v2" in model_name:
        print('*************Loading dave2v2*************')
        model = DAVE2v2()
        weights = model.load(path='./models/log/DAVE2_v2_center/Aug=True/weights_5000_best.pth')
        model.load_state_dict(weights)
        #args.eps_step = 0.001
    elif "dave2v3" in model_name:
        print('*************Loading dave2v3*************')
        model = DAVE2v3()
        weights = model.load(path="./models/log/DAVE2_v3_center/Aug=True/weights_5000_best.pth")
        model.load_state_dict(weights)
        #args.eps_step = 0.001
    elif "epoch" in model_name:
        print('*************Loading Epoch*************')
        model = Epoch()
        weights = model.load(path="./models/log/epoch/Aug=True/weights_10000_best.pth")
        model.load_state_dict(weights)

    return model
def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==True, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def get_num_non_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==False, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = False