import os 
import numpy as np 

import torch
from torch.utils.data import Dataset

from PIL import Image
from matplotlib import cm

# /mnt/data/Doutorado/dataset/tattoo/annotations
# /mnt/data/Doutorado/dataset/tattoo/images

class TattooDataset(Dataset):
    
    def __init__(self, root, split="train", test_size=0.3, img_size=None, augmentation=None, ignore_classes=True):
        
        self.root = root
        self.split = split
        self.test_size = test_size
        self.images_dir = os.path.join(self.root, 'images')
        self.targets_dir = os.path.join(self.root, 'mask_ids')
        self.images = []
        self.targets = []
        self.img_size = img_size
        self.augmentation = augmentation
        self.ignore_classes = ignore_classes
        
        if self.ignore_classes is False:
            self.class_names = ["background", "anchor", "bird", "branch", "butterfly", "cat", "crown", "diamond", "dog", "eagle", "fire", "fish", "flower", "fox", "gun", "heart", "key", "knife", "leaf", "lion", "mermaid", "octopus", "owl", "ribbon", "rope", "scorpion", "shark", "shield", "skull", "snake", "spide", "star", "tiger", "water", "wolf"]
        else:
            self.class_names = ["background", "anchor", "bird", "branch", "butterfly", "cat", "crown", "diamond", "dog", "eagle", "fire", "fish", "flower", "fox", "gun", "heart", "key", "knife", "leaf", "lion", "mermaid", "octopus", "owl", "ribbon", "rope", "scorpion", "shark", "shield", "skull", "snake", "spide", "star", "tiger", "water", "wolf"]
        self.num_classes = len(self.class_names)
        self.class_index = [i for i in range(self.num_classes)]
        
        self.palette =   [
                            (  0,	  0,	0  ), # background
                            (  0,	128,	128), # anchor
                            (192,	128,	64 ), # bird
                            (255,	192,	128), # branch
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
                            (128,	255,	128), # rope
                            (255,	128,    128), # scorpion
                            (255,	192,	0  ), # shark
                            ( 64,	192,	0  ), # shield
                            (  0,	192,	255), # skull
                            (128,	  0,	0  ), # snake
                            (128,	 64,	64 ), # spide
                            ( 64,	 64,	64 ), # star
                            (192,	  0,    128), # tiger
                            ( 64,	 64,	255), # water
                            (128,	128,	0  )] # wolf
                            # (255, 255, 255) ]   # unknown
        
        # get all images and targets
        for file_target, file_image in zip(self.targets_dir, self.images_dir):

            self.targets.append(os.path.join(self.targets_dir, file_target))
            self.images.append(os.path.join(self.images_dir, file_image))
            # print(os.path.join(target_path, file_target), os.path.join(image_path, file_image))
        
        # split the dataset
        # images_train, images_test, targets_train, targets_test = train_test_split(self.images, self.targets, test_size=self.test_size, random_state=666)

        # if self.split == "train":
        #     self.images = images_train
        #     self.targets = targets_train
        # else:
        #     self.images = images_test
        #     self.targets = targets_test
        
        print(len(self.images), len(self.targets))
        
    def __getitem__(self, index):
        
        # load image and target
        image = Image.open(self.images[index], mode='RGB')
        # target = np.load(self.targets[index])
        target = Image.open(self.targets[index], mode='L')
        
        # image name
        image_name = self.images[index].split('/')[-1]
        
        if self.img_size is not None:
            image = image.resize(self.img_size, Image.BILINEAR)
            target = Image.fromarray(target).resize(self.img_size, Image.NEAREST)
        
        # PIL Image to np.array
        image = np.array(image).astype(np.uint8)
        target = np.array(target).astype(np.uint8)
        
        # if self.ignore_classes is True: 
        #     target[target == 2] = 255  # flecha
        #     target[target == 6] = 255  # flor
        #     target[target == 7] = 255  # bandeira
        #     target[target == 8] = 255  # caveira
        #     target[target == 9] = 255  # cobra
        #     target[target == 10] = 255  # folha
        #     target[target == 11] = 255  # coroa
        #     target[target == 12] = 255  # espada
        #     target[target == 13] = 255  # sol
        #     target[target == 14] = 255  # pena
        #     target[target == 15] = 255  # passaro
        #     target[target == 17] = 255  # oculos
        #     target[target == 18] = 255  # relogio
        #     # update index
        #     target[target == 3] = 2  # escudo
        #     target[target == 4] = 3  # galho
        #     target[target == 5] = 4  # faixa
        #     target[target == 16] = 5  # coruja
        #     target[target == 19] = 6  # estrela
        #     target[target == 20] = 7  # arma
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=target)
            image, target = sample['image'], sample['mask']
        
        # normalize
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        # transforme to torch tensor
        image = torch.from_numpy(image) # (shape: (3, width, height))
        target = torch.from_numpy(target) # (shape: (width, height))
        
        # define types
        image, target = image.float(), target.long()
        
        ## torch, torch, image_name
        return image, target, image_name
            
    def __len__(self):
        return len(self.images)

    def _class_to_idx(self, dictionary=False):
        if dictionary:
            return dict(zip(self.class_names, range(self.num_classes)))
        return list(zip(self.class_names, range(self.num_classes)))

    def _get_classes_name(self):
        return self.class_names
    
    def _get_classes_index(self):
        return [x for x in range(len(self.class_names))]
    
    def _get_num_known_classes(self):
        return len(self.class_names)
    
    def _get_num_unknown_classes(self):
        return 0
    
    def _get_images_labels(self):
        return self.images, self.targets
    
    def _get_images_index(self):
        return [x for x in range(len(self.images))]  
    
    def _get_colors(self):
        return np.array(self.palette)            

def main():
    root = "/mnt/code/Doutorado/code/segmentation/semantic/prototypical-triplet-tattoo-openset-segmentation/dataset/tattoo"
    data = TattooDataset(root=root)
    print(data.num_classes)
    print(data._class_to_idx())
    
if __name__ == '__main__':
    main()

                
        
        
        
        
        