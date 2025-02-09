import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import random 
import numpy as np

def get_training_augmentation(args):
    
    width = args.input_width
    height = args.input_height
    p = args.prob_augmentation
    
    train_transform = [
        
        # Pre processing
        albu.OneOf([
            albu.Sequential([
                albu.PadIfNeeded(min_height=height, min_width=width, border_mode=0, mask_value=255, always_apply=True),
                albu.RandomResizedCrop(height=height, width=width, scale=(0.1, 0.9), ratio=(0.75, 1.25), always_apply=True),
            ]),
            albu.Sequential([
                albu.PadIfNeeded(min_height=int(height*1.5), min_width=int(width*1.5), border_mode=0, mask_value=255, always_apply=True),
                albu.CenterCrop(height=int(height*1.5), width=int(width*1.5), always_apply=True),
                albu.Resize(height=height, width=width, always_apply=True)
            ]),
            albu.Resize(height=height, width=width, always_apply=True)
        ], p=1),
        
        # Geometry transformations
        albu.OneOf([
            albu.HorizontalFlip(p=p),
            albu.VerticalFlip(p=p),
            albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, shift_limit=0.2, p=p, border_mode=0, mask_value=255),
            # albu.RandomRotate90(p=p),
            albu.Transpose(p=p)            
        ], p=1),
        
        # # # Semantic Tattoo Augmentation
        SemanticAug(transforms=[
            # albu.ChannelShuffle(always_apply=True), 
            albu.RGBShift(always_apply=True, r_shift_limit=128, g_shift_limit=128, b_shift_limit=128), 
            albu.ToGray(always_apply=True),
            albu.RandomToneCurve(always_apply=True, scale=0.75),
            albu.NoOp()
        ]),
                
        # # # Image transformations
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=p),
            # albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=p),
            albu.GaussianBlur(blur_limit=(5), p=p),
            albu.FancyPCA(p=p),
        ], p=1),
        
        # # Dropout transformations
        albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=p, mask_fill_value=255),
        
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation(args):
    
    width = args.input_width
    height = args.input_height
    
    """Add paddings to make image shape divisible by 32"""
    val_transform = [
        albu.Resize(height=height, width=width, always_apply=True),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return albu.Compose(val_transform)

def get_view(args):
    
    width = args.input_width
    height = args.input_height
    
    """Add paddings to make image shape divisible by 32"""
    val_transform = [
        albu.Resize(height=height, width=width, always_apply=True),
        ToTensorV2()
    ]
    return albu.Compose(val_transform) 

def get_view_numpy(args):
    
    width = args.input_width
    height = args.input_height
    
    """Add paddings to make image shape divisible by 32"""
    val_transform = [
        albu.Resize(height=height, width=width, always_apply=True),
    ]
    return albu.Compose(val_transform) 

class SemanticAug(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, transforms=[albu.NoOp()], p=None, ignore_classes=[0,255]):
        self.transforms = transforms
        self.p = p
        self.ignore_classes = ignore_classes

    def __call__(self, image, mask):
        
        semantic_ids = np.unique(mask)

        for i in self.ignore_classes:
            semantic_ids = np.delete(semantic_ids, np.where(semantic_ids == i))

        for id in semantic_ids:
                        
            semantic_mask = np.zeros(mask.shape, dtype="uint8")
            semantic_mask[mask == id] = 1
            
            id_trans = random.choices(np.arange(len(self.transforms)), self.p)[0]
            transformed_image = self.transforms[id_trans](image=image,mask=mask)['image']
            image[semantic_mask == 1] = transformed_image[semantic_mask == 1]

        return {'image': image, 'mask': mask}