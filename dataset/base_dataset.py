import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def open_preprocessing_tattoo(args, image_path, segm_path):
    img = Image.open(image_path).convert('RGB')
    segm = np.load(segm_path)
    if args.custom_classes:
        segm = encode_segmap_tattoo(args, segm)
    segm = remove_open_class(args.split, segm, args.original_num_classes, args.openset_idx)
    return np.array(img), np.array(segm)
    
def remove_open_class(split, segm, num_class, open_idx):
    
    closeset_idx = [i for i in range(0, num_class)]
    for i in open_idx:
        closeset_idx.remove(i)
        segm[segm == i] = 128 # tmp ids
        
    class_map = dict(zip(closeset_idx, range(len(closeset_idx))))
    
    if split == 'train':
        ignore_index = 255 # closeset_idx
    else:
        ignore_index = num_class - (len(open_idx))
        
    class_map[128] = ignore_index
    closeset_idx.append(128)
        
    # somente este precisa ser excutado em toda imagem
    for i in closeset_idx:
        segm[segm == i] = class_map[i]
    
    return segm

def encode_segmap_tattoo(args, mask):
        
    # class_names = ["background", "anchor", "bird", "branch", "butterfly", "cat", "crown", "diamond", "dog", "eagle", "fire", "fish", "flower", "fox", "gun", "heart", "key", "knife", "leaf", "lion", "mermaid", "octopus", "owl", "ribbon", "rope", "scorpion", "shark", "shield", "skull", "snake", "spide", "star", "tiger", "water", "wolf"]
       
    # class_map
    class_map = {}
    class_map[0] = 0  # background
    class_map[1] = 1  # anchor
    class_map[2] = 2  # bird
    class_map[3] = 128  # branch (background)
    class_map[4] = 3  # butterfly
    class_map[5] = 4  # cat
    class_map[6] = 5  # crown
    class_map[7] = 6  # diamond
    class_map[8] = 7  # dog
    class_map[9] = 8  # eagle
    class_map[10] = 9  # fire
    class_map[11] = 10  # fish
    class_map[12] = 11  # flower 
    class_map[13] = 12  # fox
    class_map[14] = 13  # gun
    class_map[15] = 14  # heart
    class_map[16] = 15  # key
    class_map[17] = 16  # knife
    class_map[18] = 17  # leaf 
    class_map[19] = 18  # lion
    class_map[20] = 19  # mermaid
    class_map[21] = 20  # octopus
    class_map[22] = 21  # owl
    class_map[23] = 22  # ribbon
    class_map[24] = 128  # rope 
    class_map[25] = 23  # scorpion
    class_map[26] = 24  # shark
    class_map[27] = 25  # shield
    class_map[28] = 26  # skull
    class_map[29] = 27  # snake
    class_map[30] = 28  # spider
    class_map[31] = 29  # star
    class_map[32] = 30  # tiger
    class_map[33] = 31  # water
    class_map[34] = 32  # wolf
    if args.split == 'train':
        class_map[35] = 255  # unknown
    else:
        class_map[35] = 33  # unknown
    class_map[128] = 0  # to background

    # original index
    class_index = range(0, 36)
    
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
    
    # x1 = np.unique(mask)
    
    # for c in class_index:
    for c in class_index:  
        new_mask[c == mask] = class_map[c]
    
    # 128 to background
    new_mask[128 == new_mask] = class_map[128]
    
    # map to unknown class
    new_mask = np.squeeze(new_mask)
            
    # x2 = np.unique(new_mask)
    # print(x1, x2)
    
    return new_mask

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, odgt, augmentation, **kwargs):
        
        self.args = args
        self.augmentation = augmentation
        self.segm_downsampling_rate = 8
        
        self.target = []
        self.images = []
        self.indexes = []
        
        # update indexes and class names
        self.tmp_names = self._get_categories()
        self.args.original_num_classes = len(self.tmp_names)
        self.class_names = []
        for i in range(0, len(self.tmp_names)):
            if i not in args.openset_idx:
                self.class_names.append(self.tmp_names[i])
        
        # update num_classes
        self.num_classes = len(self.class_names)
        self.class_index = [i for i in range(self.num_classes)]
        
        # include unknown class in test
        if self.args.split == "test": 
            #open-set segmentation
            self.class_names.append("unknown")
        
        # get path from odgt
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')][0]

        for i, s in enumerate(self.list_sample):
            self.indexes.append(i)
            self.images.append(s['fpath_img'])
            self.target.append(s['fpath_segm'])
            
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
    
    def __getitem__(self, index):
        
        image_name = self.images[index].split('/')[-1]
        
        if self.args.dataset == 'tattoo':
            image, target = open_preprocessing_tattoo(self.args, self.images[index], self.target[index])
        else:
            print('x')
        
        # apply augmentation
        if self.augmentation != None:
            sample = self.augmentation(image=image, mask=target)
            image, target = sample['image'], sample['mask']

        return image, target, image_name
    
    def __len__(self):
        return self.num_sample
    
    def _class_to_idx(self, dictionary=False):
        if dictionary:
            return dict(zip(self.class_names, range(self.num_classes)))
        return list(zip(self.class_names, range(self.num_classes)))
    
    def _get_images_index(self):
        return self.indexes
    
    def _get_images_labels(self):
        return self.images, self.targets
    
    def _get_classes_index(self):
        return [x for x in range(len(self.class_names))]
    
    def _get_classes_names(self):
        return self.class_names
    
    def _get_num_known_classes(self):
        if self.args.split == 'train':
            return len(self.class_names) 
        else:
            # return len(self.class_names) - len(self.args.openset_idx)
            return len(self.class_names) - 1
    
    def _get_num_unknown_classes(self):
        if len(self.args.openset_idx) == 0:
            return 1
        else:
            return len(self.args.openset_idx)
    
    def _get_categories(self):
        
        if self.args.dataset == 'tattoo':
            if self.args.custom_classes:
                class_names = ["background", "anchor", "bird", "butterfly", "cat", "crown", "diamond", "dog", "eagle", "fire", "fish", "flower", "fox", "gun", "heart", "key", "knife", "leaf", "lion", "mermaid", "octopus", "owl", "ribbon", "scorpion", "shark", "shield", "skull", "snake", "spide", "star", "tiger", "water", "wolf"]
            else:
                class_names = ["background", "anchor", "bird", "branch", "butterfly", "cat", "crown", "diamond", "dog", "eagle", "fire", "fish", "flower", "fox", "gun", "heart", "key", "knife", "leaf", "lion", "mermaid", "octopus", "owl", "ribbon", "rope", "scorpion", "shark", "shield", "skull", "snake", "spide", "star", "tiger", "water", "wolf"]
        else: 
            raise NotImplementedError(f'This ({self.args.dataset}) dataset not supported')
        
        return class_names
    
    def _get_colors(self):
        
        if self.args.dataset == 'tattoo':
            palette = [
                            (255,	255,	255), # background
                            (  0,	128,	128), # anchor
                            (192,	128,	64 ), # bird
                            # (255,	192,	128), # branch
                            (192,	  0,	128), # butterfly
                            (192,	255,	64 ), # cat
                            ( 64,	128,	0  ), # crown
                            (128,	192,	255), # diamond
                            (192,	 64,	0  ), # dog
                            (  0,	192,	192), # eagle
                            (128,	128,	128), # fire
                            (128,	  0,	192), # fish
                            (255,	128,	255), # flower
                            (192,	192,	64 ), # fox
                            ( 64,	255,	128), # gun
                            (192,	128,	0  ), # heart
                            (128,	192,	192), # key
                            (  0,   128,	255), # knife
                            ( 64,   255,	64 ), # leaf
                            (192,	128,	255), # lion
                            (255,	192,	64 ), # mermaid
                            (  0,	192,	0  ), # octopus
                            (192,	  0,	0  ), # owl
                            (  0,	  0,	192), # ribbon
                            # (128,	255,	128), # rope
                            (255,	128,    128), # scorpion
                            (255,	192,	0  ), # shark
                            ( 64,	192,	0  ), # shield
                            (  0,	192,	255), # skull
                            (128,	  0,	0  ), # snake
                            (128,	 64,	64 ), # spide
                            ( 64,	 64,	64 ), # star
                            (192,	  0,    128), # tiger
                            ( 64,	 64,	255), # water
                            (128,	128,	0  ), # wolf
                            (  0,     0,    0  )] # unknown
        else: 
            raise NotImplementedError(f'This ({self.args.dataset}) dataset not supported')
            
        colors = palette.copy()

        #remove openset class
        for i in self.args.openset_idx:
            del colors[i]
            
        return np.array(colors)
    
    def get_cls_num_list():
    
        cls_num_list = [       
            56186676, 
            380819,
            426069,
            416534,
            608116,
            409481,
            126316,
            463750,
            847624,
            132463,
            470082,
            600377,
            304439,
            363380,
            353420,
            159020,
            249427,
            250721,
            723456,
            420553,
            686082,
            675385,
            459177,
            578642,
            363646,
            235122,
            858219,
            463464,
            191459,
            103977,
            869007,
            325885,
            744316
        ]
        
        return cls_num_list