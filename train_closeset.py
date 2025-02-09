import sys

import torch
torch.set_float32_matmul_precision('medium')
# print('Is cuda available?:', torch.cuda.is_available())

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# cuDNN
import ctypes
if sys.platform in ('linux2', 'linux'):
    # _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.6', 'libcudnn.so.6.0.21']
    _libcudnn_libname_list = ['libcudnn.so']
else:
    raise RuntimeError('unsupported platform')

_libcudnn = None
for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break
if _libcudnn is None:
    raise OSError('cuDNN library not found')

print("cuDNN Version.: " + str(_libcudnn.cudnnGetVersion()))
print("cuDNN Version.: " + str(torch.backends.cudnn.version()))

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import json
import numpy as np

from parse_args import parse_args_train_closeset
from utils.func_preparation import reproducibility, make_outputs_dir, create_dir
from utils.func_preparation import get_dataset_train, get_model, get_loss, get_callbacks
from utils.func_preparation import init_history, save_plot_history, save_best_model_history, save_model, early_stopping_loss, early_stopping_metric

from utils.augmentations import get_training_augmentation, get_validation_augmentation
# from utils.trainning import TrainEpoch, ValidEpoch

import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin

from models.seg_module_base import SegmentationModuleBase

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold, train_test_split

def train_model_cross(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    
    # get dataset
    train_dataset, valid_dataset = get_dataset_train(args, get_training_augmentation, get_validation_augmentation)
    
    # info dataset
    all_index = train_dataset._get_images_index()
    num_classes = train_dataset._get_num_known_classes()
    args.num_classes = num_classes
    
    print('Known class.:', num_classes)
    print(train_dataset._class_to_idx())

    # tmp
    savedir = args.savedir
    
    # save args
    with open(os.path.join(args.savedir, 'args.json'), "w") as write_file:
        json.dump(vars(args), write_file)
    
    # cross validation kfolds
    if args.train_type == 'cross':
        Kfold = KFold(n_splits=args.k_fold, shuffle=False)
        print('Num. K-folds.:', args.k_fold)
    
        for k, (train_index, test_index) in enumerate(Kfold.split(all_index)):
            
            print("Training K-fold.: ", k)
            
            # update savedir
            args.savedir = os.path.join(savedir, str(k))
            create_dir(args.savedir)
            
            # update parans to train 
            model = get_model(args)    
            loss = get_loss(args)
            callbacks = get_callbacks(args)
            
            train_sampler = SubsetRandomSampler(train_index)
            valid_sampler = SubsetRandomSampler(test_index)

            model = SegmentationModuleBase(
                args=args,
                model=model,
                loss=loss,
                batch_size=args.batch_size,
                trainset=train_dataset,
                train_sampler=train_sampler,
                valset=valid_dataset,
                valid_sampler=valid_sampler,
                num_classes=num_classes
            )
            model.cuda()
            
            trainer = pl.Trainer(
                        accelerator='cuda', # 
                        max_epochs=args.max_epochs,
                        min_epochs=1,
                        precision=16, 
                        callbacks=callbacks,
                        num_sanity_val_steps=0,
                        logger=False,
                        log_every_n_steps=1
                    )
            
            # training
            trainer.fit(model)  
    else: # holdout
        # update savedir
        args.savedir = os.path.join(savedir, 'holdout')
        create_dir(args.savedir)
        
        # update parans to train 
        model = get_model(args)    
        loss = get_loss(args)
        callbacks = get_callbacks(args)
        
        model = SegmentationModuleBase(
            args=args,
            model=model,
            loss=loss,
            batch_size=args.batch_size,
            trainset=train_dataset,
            train_sampler=None,
            valset=valid_dataset,
            valid_sampler=None,
            num_classes=num_classes
        )
        model.cuda()
        
        trainer = pl.Trainer(
                    accelerator='cuda', # 
                    max_epochs=args.max_epochs,
                    min_epochs=1,
                    precision=16, 
                    callbacks=callbacks,
                    num_sanity_val_steps=0,
                    logger=False,
                    log_every_n_steps=1
                )
        
        # training
        trainer.fit(model)  
            
    # root savedir
    args.savedir = savedir

if __name__ == '__main__':
    
    # get params
    args = parse_args_train_closeset()

    # reproducibility
    reproducibility(args)
    
    # create new directory to save outputs
    make_outputs_dir(args)
    
    # training the model
    train_model_cross(args)
    
    # save args
    with open(os.path.join(args.savedir, 'args.json'), "w") as write_file:
        json.dump(vars(args), write_file)
