import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # gpu_id to use

import json
import numpy as np
import pandas as pd 

from tqdm import tqdm
from skimage import util

import torch
from torch.utils.data import DataLoader

from parse_args import parse_args_train_report

from utils.func_preparation import reproducibility, create_dir
from utils.func_preparation import compute_iou, get_dataset, save_plot_confusion_matrix, save_closeset_predictions, generete_report
from utils.augmentations import get_validation_augmentation, get_view

def compute_predictions_closetset(args):
    
    # get dataset and dataloader
    train_dataset, valid_dataset = get_dataset(args, get_view, get_validation_augmentation)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # info data
    categories = valid_dataset._get_classes_name()
    class_index = valid_dataset._get_classes_index()
    colors = valid_dataset._get_colors()
    
    # load trained model
    model = torch.load(args.savedir + '/best_overall_model.pth')
    model.to(args.device)
    model.eval()
    
    # predictions and labels
    predictions, labels = [], []

    for t, v in tqdm(zip(train_loader, valid_loader), total=len(train_dataset), desc='Compute Predictions Report'):
            
        # get batchs
        x_view, _, image_name = t
        x, y, _ = v

        x, y = x.to(args.device), y.to(args.device)
        
        # multi lib models
        y_pred = model(x)
            
        # predictions to softmax and reshape
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        
        # save images predictions
        if args.save_images:
                
            image = util.img_as_ubyte(x_view.permute(0, 2, 3, 1).detach().cpu().squeeze().numpy())
            label = y.cpu().numpy().squeeze().astype(np.uint8)
            prediction = y_pred.cpu().numpy().squeeze().astype(np.uint8)

            # ISPRS dataset apply black pixel to ignored pixels

            image_name = str(image_name[0])

            save_closeset_predictions(args, image, label, prediction, class_index, image_name, colors)
            
        # torch tensor to numpy
        y = y.detach().cpu()
        y_pred = y_pred.detach().cpu()
        
        predictions.append(y_pred)
        labels.append(y)

    # reshape
    predictions = torch.cat(predictions, dim=1)
    labels = torch.cat(labels, dim=1)

    # reshape labels and predictions (to numpy)    
    labels = torch.squeeze(labels.reshape((labels.size()[0] * labels.size()[1] * labels.size()[2])))
    predictions = torch.squeeze(predictions.reshape((predictions.size()[0] * predictions.size()[1] * predictions.size()[2]))).numpy()
            
    return predictions, labels, categories

if __name__ == '__main__':
    
    args, parser = parse_args_train_report()
    
    # load args folder for eval
    args_load = os.path.join(args.folder_path, args.folder_id, 'args.json')
    
    with open(args_load, 'rt') as f:
        args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=args)

    # reproducibility
    reproducibility(args)
    
    # start
    predictions, labels, categories = compute_predictions_closetset(args)
    generete_report(args, predictions, labels, categories, 'closeset')