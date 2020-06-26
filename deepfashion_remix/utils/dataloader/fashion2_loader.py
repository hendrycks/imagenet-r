import os
import collections
import json

import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import collections
import torch
import torchvision

from torch.utils import data
from tqdm import tqdm
import random 

from scipy.io import loadmat


def get_annotations(
        root, split, n_classes, scales, 
        occlusion, zoom, viewpoint, negate=(False,False,False,False), load=False):
    root_path = os.path.join(root,split,"annos")
    if load:
        annos = np.load("annos.npy")
        mask = np.load("mask.npy")
        return annos, mask
    if (type(scales) == int):
        scales = [scales]
    if (type(zoom) == int):
        zoom = [zoom]
    if (type(viewpoint) == int):
        viewpoint = [viewpoint]
    if (type(occlusion) == int):
        occlusion = [occlusion]
    anno_files = sorted(os.listdir(root_path))
    annos = np.zeros((len(anno_files), n_classes))
    mask = np.zeros((len(anno_files)), dtype=bool)

    for index, anno_f in enumerate(tqdm(anno_files)):
        with open(os.path.join(root_path,anno_f)) as f:
            anno = json.load(f)
        #anno is a dict with source, pairid, and item1, item2, etc
        for item in anno:
            # the other two are source and pairid
            item_found = True
            if "item" in item:
                if (anno[item]["occlusion"] in occlusion):
                    item_found = item_found and not negate[0]
                else: 
                    item_found = item_found and negate[0]
                if (anno[item]["scale"] in scales):
                    item_found = item_found and not negate[1]
                else: 
                    item_found = item_found and negate[1]
	
                if (anno[item]["viewpoint"] in viewpoint):
                    item_found = item_found and not negate[2]
                else: 
                    item_found = item_found and negate[2]
	
                if (anno[item]["zoom_in"] in zoom):
                    item_found = item_found and not negate[3]
                else: 
                    item_found = item_found and negate[3]
	
                if item_found:
                    mask[index] = True
                    annos[index][anno[item]["category_id"] - 1] = 1

    #np.save("annos",annos)
    #np.save("mask", mask)
    return annos, mask

class fashion2loader(data.Dataset):
    def __init__(self,root="../", label_root="../", split="train", 
            transform=None, label_transform=None,
            scales=(2), occlusion=(2), zoom=(1), viewpoint=(2), negate=(0,0,0,0),
            load=False):
        self.root = root
        self.split = split
        self.n_classes = 13
        self.transform = transform
        self.label_transform = label_transform
        tmp_path = os.path.join(root,split,"image")
        self.Imglist = sorted(os.listdir(tmp_path))
        for i in range(len(self.Imglist)):
            self.Imglist[i] = os.path.join(root,split,"image", self.Imglist[i])
        self.Imglist = np.asarray(self.Imglist)
        self.GT, self.mask  = get_annotations(
                label_root, split, self.n_classes, scales, 
                occlusion, zoom, viewpoint, negate=negate, load=load)
        self.Imglist = self.Imglist[self.mask]
        self.GT = self.GT[self.mask]


    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):

        img = Image.open(self.Imglist[index]).convert('RGB')
        if self.split == "test":
            lbl = np.zeros(self.n_classes)
        else:
            lbl = self.GT[index]

        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.transform is not None:
            img_o = self.transform(img)
            # img_h = self.img_transform(self.h_flip(img))
            # img_v = self.img_transform(self.v_flip(img))
            imgs = img_o
        else:
            imgs = img
        random.seed(seed)
        if self.label_transform is not None:
            label_o = self.label_transform(lbl)
            # label_h = self.label_transform(self.h_flip(label))
            # label_v = self.label_transform(self.v_flip(label))
            lbls = label_o
        else:
            lbls = lbl

        return [imgs], lbls


