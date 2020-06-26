import argparse
import itertools
import numpy as np
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from augmix import AugMix
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from utils.dataloader.fashion2_loader import *

from tqdm import tqdm
import torchvision.transforms as transforms
#---- own transformations
from utils.transform import ReLabel, ToLabel, ToSP, Scale

import model.bit_models as bit_models
from model.se_resnet import se_resnet50 
from model.classifiersimple import clssimp

def speckle_noise_torch(data):
    """samples speckle noise according to the list stds, adds it to data
       and returns the noisy data.
    """
    stds = [.15, .2, 0.35, 0.45, 0.6]
    c = np.random.choice(stds, data.shape[0], replace=True)
    noise = torch.empty(data.shape, device=data.device).normal_() * torch.Tensor(c).view(-1, 1, 1).to(data.device)
    scaled_noise = data * noise

    #assert (scaled_noise.shape == data.shape), "Shape of scaled speckle noise does not equal the shape of the input!"
				        
    return torch.clamp(data + scaled_noise, 0, 1)

def train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.augmix:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((args.img_size),scale=(0.5, 2.0)),
        ])  
    elif args.speckle:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((args.img_size),scale=(0.5, 2.0)),
            transforms.ToTensor(),
            transforms.RandomApply([transforms.Lambda(lambda x: speckle_noise_torch(x))], p=0.5),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((args.img_size),scale=(0.5, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])
    if args.cutout:
        train_transform.transforms.append(transforms.RandomErasing())

    val_transform = transforms.Compose([
            transforms.Scale((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ])


    label_transform = transforms.Compose([
            ToLabel(),
        ])
    print("Loading Data")
    if args.dataset == "deepfashion2":
        loader = fashion2loader(
            "../",
            transform = train_transform,
            label_transform = label_transform,
            #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
            scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
            negate=args.negate,
            #load=True,
            )
        if args.augmix:
            loader = AugMix(loader, args.augmix)
        if args.stylize:
            style_loader = fashion2loader(
                root="../../stylize-datasets/output/",
                transform = train_transform,
                label_transform = label_transform,
                #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
                scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
                negate=args.negate,
                #load=True,
                )
            loader = torch.utils.data.ConcatDataset([loader, style_loader])
        valloader = fashion2loader(
            "../",
            split="validation",
            transform = val_transform,
            label_transform = label_transform,
            #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
            scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
            negate=args.negate,
            )
    elif args.dataset == "deepaugment":
        loader = fashion2loader(
            "../",
            transform = train_transform,
            label_transform = label_transform,
            #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
            scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
            negate=args.negate,
            #load=True,
            )  
        loader1 = fashion2loader(
            root="../../deepaugment/EDSR/",
            transform = train_transform,
            label_transform = label_transform,
            #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
            scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
            negate=args.negate,
            #load=True,
            )
        loader2 = fashion2loader(
            root="../../deepaugment/CAE/",
            transform = train_transform,
            label_transform = label_transform,
            #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
            scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
            negate=args.negate,
            #load=True,
            )
        loader = torch.utils.data.ConcatDataset([loader, loader1, loader2])
        if args.augmix:
            loader = AugMix(loader, args.augmix)
        if args.stylize:
            style_loader = fashion2loader(
                root="../../stylize-datasets/output/",
                transform = train_transform,
                label_transform = label_transform,
                #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
                scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
                negate=args.negate,
                #load=True,
                )
            loader = torch.utils.data.ConcatDataset([loader, style_loader])
        valloader = fashion2loader(
            "../",
            split="validation",
            transform = val_transform,
            label_transform = label_transform,
            #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=(True,True,True,True),
            scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
            negate=args.negate,
            )
 
    else:
        raise AssertionError
    print("Loading Done")

    n_classes = args.num_classes
    train_loader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)


    print("number of images = ", len(train_loader))
    print("number of classes = ", n_classes)


    print("Loading arch = ", args.arch)
    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
    elif args.arch == "resnet50":
        orig_resnet = torchvision.models.resnet50(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
    elif args.arch == "resnet152":
        orig_resnet = torchvision.models.resnet152(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
    elif args.arch == "se":
        model = se_resnet50(pretrained=True)
        features = list(model.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
    elif args.arch == "BiT-M-R50x1":
        model = bit_models.KNOWN_MODELS[args.arch](head_size=2048, zero_head=True)
        model.load_from(np.load(f"{args.arch}.npz"))
        features = list(model.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
    elif args.arch == "BiT-M-R101x1":
        model = bit_models.KNOWN_MODELS[args.arch](head_size=2048, zero_head=True)
        model.load_from(np.load(f"{args.arch}.npz"))
        features = list(model.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
 

    if args.load == 1:
        model.load_state_dict(torch.load(args.save_dir + args.arch + str(args.disc) +  ".pth"))
        clsfier.load_state_dict(torch.load(args.save_dir + args.arch +"clssegsimp" + str(args.disc) +  ".pth"))

    gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_dataparallel = len(gpu_ids) > 1
    print("using data parallel = ", use_dataparallel, device, gpu_ids)
    if use_dataparallel:
        gpu_ids = [int(x) for x in range(len(gpu_ids))]
        model = nn.DataParallel(model, device_ids=gpu_ids)
        clsfier = nn.DataParallel(clsfier, device_ids=gpu_ids)
    model.to(device)
    clsfier.to(device)

    if args.finetune:
        if args.opt == "adam":
            optimizer = torch.optim.Adam([{'params': clsfier.parameters()}], lr=args.lr)
        else:
            optimizer = torch.optim.SGD(clsfier.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    else:
        if args.opt == "adam":
            optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.lr/10},{'params': clsfier.parameters()}], lr=args.lr)
        else:
            optimizer = torch.optim.SGD(itertools.chain(model.parameters(), clsfier.parameters()), 
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                   1 + np.cos(step / total_steps * np.pi))

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
               optimizer,
               lr_lambda=lambda step: cosine_annealing(
                   step,
                   args.n_epochs * len(train_loader),
                   1,  # since lr_lambda computes multiplicative factor
                   1e-6 / (args.lr * args.batch_size / 256.)))

    bceloss = nn.BCEWithLogitsLoss()
    for epoch in range(args.n_epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            if args.augmix:
                x_mix1, x_orig = images
                images = torch.cat((x_mix1, x_orig), 0).to(device)
            else: 
                images = images[0].to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            
            outputs = model(images)
            outputs = clsfier(outputs)
            if args.augmix:
                l_mix1, outputs = torch.split(outputs, x_orig.size(0))

            if args.loss == "bce":
                if args.augmix:
                    if random.random() > 0.5:
                        loss = bceloss(outputs, labels)
                    else:
                        loss = bceloss(l_mix1, labels)
                else:
                    loss = bceloss(outputs, labels)
            else:
                print("Invalid loss please use --loss bce")
                exit()

            loss.backward()
            optimizer.step()
            if args.use_scheduler:
                scheduler.step()

        print(len(train_loader))
        print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epochs, loss.data))

        save_root = os.path.join(args.save_dir,args.arch)
        if not os.path.exists(save_root):
	    os.makedirs(save_root)
        if use_dataparallel:
            torch.save(model.module.state_dict(), os.path.join(save_root, str(args.disc) + ".pth"))
            torch.save(clsfier.module.state_dict(), os.path.join(save_root, "clssegsimp" + str(args.disc) + ".pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_root, str(args.disc) +  ".pth"))
            torch.save(clsfier.state_dict(), os.path.join(save_root, 'clssegsimp' + str(args.disc) +  ".pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='Architecture to use [\'resnet50, resnet101, resnet152, se, BiT-M-R50x1, BiT-M-R101x1\']')
    parser.add_argument('--dataset', type=str, default='deepfashion2', 
                        help='Dataset to use [\'deepfashion2, deepaugment\']')
    parser.add_argument('--opt', type=str, default='adam',
                        help='Optimizer to use [\'adam, sgd\']')
    parser.add_argument('--loss', type=str, default='bce',
                        help='Loss to use only [\'bce\']')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--num_classes', type=int, default=13,
                        help='number of workers')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('-e','--n_epochs', type=int, default=80, 
                        help='# of the epochs')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='starting at what epoch')
    parser.add_argument('-b','--batch_size', type=int, default=100, 
                        help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.9,   
                        help='Learning Rate Momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,   
                        help='Weight Decay')
    parser.add_argument('--scales', nargs='+', default=[2], type=int)
    parser.add_argument('-occ', "--occlusion", nargs='+', default=[2], type=int)
    parser.add_argument('--zoom', nargs='+', default=[1], type=int)
    parser.add_argument('-vp', "--viewpoint", nargs='+', default=[2], type=int)
    parser.add_argument('--negate', nargs='+', default=[False,False,False,False], type=int,
        help='to negate occlusion, scales, viewpoint, and zoom respectively, passed in as 0, 1')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--augmix', default=None, type=int, help="specify augmix severity.")
    parser.add_argument('--speckle', action='store_true', help="train with speckle augmentation.")
    parser.add_argument('--stylize', action='store_true', help="train with stylized data.")
    parser.add_argument('--cutout', action='store_true', help="train with random erasure augmentation.")
    parser.add_argument('--load', type=int)
    parser.add_argument('--disc', type=str)
    parser.add_argument("--save_dir", type=str, default="./savedmodels/")
    args = parser.parse_args()
    train(args)
