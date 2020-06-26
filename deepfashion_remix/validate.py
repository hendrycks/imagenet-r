import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from utils.dataloader.fashion2_loader import *
import random
# import tqdm
import torchvision.transforms as transforms
#---- your own transformations
from utils.transform import ReLabel, ToLabel, ToSP, Scale

import model.bit_models as bit_models
#from model.model_resnet import ResidualNet
from model.se_resnet import se_resnet50
from model.classifiersimple import *
import torchvision
from sklearn import metrics

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def validate(args):
    # Setup Dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Scale((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    label_transform = transforms.Compose([
            ToLabel(),
            # normalize,
        ])
    if args.dataset == "deepfashion2":
        if not args.concat_data:
            valloader = fashion2loader(
      	        "../",
                split="validation",
       	        transform = val_transform,
       	       	label_transform = label_transform,
       	       	#scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1),
                scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
		negate=args.negate,
       	       	)
        else: # lets concat train and val for appropriate labels
            loader1 = fashion2loader(
                "../",
                transform = val_transform,
                label_transform = label_transform,
                #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=True,
                scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
                negate=args.negate,
                #load=True,
                )
            loader2 = fashion2loader(
                "../",
                split="validation",
                transform = val_transform,
                label_transform = label_transform, 
                #scales=(-1), occlusion=(-1), zoom=(-1), viewpoint=(-1), negate=True, 
		scales=args.scales, occlusion=args.occlusion, zoom=args.zoom, viewpoint=args.viewpoint,
                negate=args.negate,
                )
            valloader =  torch.utils.data.ConcatDataset([loader1, loader2]) 
    else:
        raise AssertionError

    n_classes = args.num_classes
    valloader = data.DataLoader(valloader, batch_size=args.batch_size, num_workers=4, shuffle=False)

    print("Number of samples = ", len(valloader))
    print("Loading arch = ", args.arch)

    if args.arch == 'resnet101':
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
    elif args.arch == 'resnet50':
        orig_resnet = torchvision.models.resnet50(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes)
    elif args.arch == 'resnet152':
        orig_resnet = torchvision.models.resnet152(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048,n_classes) 
    elif args.arch == 'se':
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

    model.load_state_dict(torch.load(args.save_dir + args.arch +"/"+ str(args.disc) +  ".pth"))
    clsfier.load_state_dict(torch.load(args.save_dir + args.arch +"/"+'clssegsimp' + str(args.disc) +  ".pth"))

    model.eval()
    clsfier.eval()

    if torch.cuda.is_available():
        model.cuda(0)
        clsfier.cuda(0)

    model.eval()
    gts = {i:[] for i in range(0,n_classes)}
    preds = {i:[] for i in range(0,n_classes)}
    # gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(valloader)):
        images = images[0].cuda()
        labels = labels.cuda().float()
        
        outputs = model(images)
        outputs = clsfier(outputs)
        outputs = F.sigmoid(outputs)
        pred = outputs.data.cpu().numpy()
        gt = labels.data.cpu().numpy()
        
        for label in range(0,n_classes):
            gts[label].extend(gt[:,label])
            preds[label].extend(pred[:,label])

    FinalMAPs = []
    for i in range(0,n_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(gts[i], preds[i]);
        FinalMAPs.append(metrics.auc(recall , precision));
    print(FinalMAPs)
    tmp = []
    for i in range(len(gts)):
        tmp.append(gts[i])
    gts = np.array(tmp)

    FinalMAPs = np.array(FinalMAPs)
    denom = gts.sum()
    gts = gts.sum(axis=-1)
    gts = gts / denom
    res = np.nan_to_num(FinalMAPs * gts)
    print((res).sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='resnet50',
        help='Architecture to use [\'fcn32s, unet, segnet etc\']')
    parser.add_argument('--dataset', type=str, default='deepfashion2',
        help='Dataset to use [\'deepfashion2\']')
    parser.add_argument('--opt', type=str, default='',
        help='Optimizer to use [\'adam, sgd\']')
    parser.add_argument('--loss', type=str, default='',
        help='Loss to use [\'bce\']')
    parser.add_argument('--num_workers', type=int, default=4,
        help='number of workers')
    parser.add_argument('--num_classes', type=int, default=13, 
        help='number of workers') 
    parser.add_argument('--img_size', type=int, default=256,
        help='Height of the input image')  
    parser.add_argument('--n_epochs', type=int, default=80,
        help='# of the epochs')  
    parser.add_argument('-b', '--batch_size', type=int, default=20,
        help='Batch Size')
    parser.add_argument('--scales', nargs='+', default=[2], type=int)
    parser.add_argument('-occ', "--occlusion", nargs='+', default=[2], type=int)
    parser.add_argument('--zoom', nargs='+', default=[1], type=int)
    parser.add_argument('-vp', "--viewpoint", nargs='+', default=[2], type=int)
    parser.add_argument('--negate', nargs='+', default=[True,True,True,True], type=int, 
        help='to negate occlusion, scales, viewpoint, and zoom respectively, passed in as 0, 1')

    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--concat_data', action='store_true', help="Whether to concatenate the train and val") 
    parser.add_argument('--load', type=int)
    parser.add_argument('--disc', type=str) 
    parser.add_argument("--save_dir", type=str, default="./savedmodels/") 
    args = parser.parse_args()
    args.negate = [bool(x) for x in args.negate]
    validate(args)
