# https://medium.com/innovation-res/simplify-your-pytorch-code-with-pytorch-lightning-5d9e4ebd3cfd
# https://medium.com/data-science-at-microsoft/how-to-smoothly-integrate-meanaverageprecision-into-your-training-loop-using-torchmetrics-7d6f2ce0a2b3
# https://pub.towardsai.net/improve-your-model-validation-with-torchmetrics-b457d3954dcd
# https://rising.readthedocs.io/en/stable/lightning_segmentation.html
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.functional import pairwise_cosine_similarity
from torchvision import transforms

import numpy as np
import torch.nn as nn
from torch.nn.functional import interpolate

# to save the time to compute the batch
import time

def user_scattered_collate(batch):
    return batch

class SegmentationModuleBase(pl.LightningModule):
    
    def __init__(self, args, model, loss, batch_size, trainset, train_sampler, valset, valid_sampler, num_classes):
        super(SegmentationModuleBase, self).__init__()
        self.args = args
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        
        # dataset to train
        self.trainset = trainset
        self.train_sampler = train_sampler
        self.valset = valset
        self.valid_sampler = valid_sampler
        # scheduler
        # define T-Max as max. epochs of train #  config to epoch
        self.t_max = args.max_epochs
        self.num_classes = num_classes
        
        # data augmentation
        # self.cutmix_or_none = transforms.RandomChoice([self.cutmix, self.cutNone], p=[0.2, 0.8])
        self.cutmix_or_none = transforms.RandomChoice([self.cutmix, self.cutNone])
        
        # define metric
        # self.jaccard_train = MulticlassJaccardIndex(ignore_index=255, num_classes=self.num_classes, average='weighted')
        self.jaccard_val = MulticlassJaccardIndex(ignore_index=255, num_classes=self.num_classes, average='weighted')
        
        # predictions, labels, and categories to plot TSNE
        if args.plot_tsne and self.trainset != None:
            self.last_X = None
            self.last_y = None 
            self.categories = self.trainset._get_classes_names() # no categories
        
    def forward(self, x, y):
        if self.args.feature_extractor:
            return self.model(x)
        else:
            if self.args.model == 'seg_former': 
                outs = self.model(pixel_values=x, labels=y)[1]
                outs = interpolate(outs, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
                return outs
            elif self.args.model == 'swin': 
                outs = self.model(x)[0]
                return outs
            else:
                return self.model(x)
    
    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    def training_step(self, batch, batch_idx):
        # forward, backward, optimization and schedulers
        if self.args.model == 'ppm_resnet50' or self.args.model == 'psp_resnet101':
            x, y = batch[0]['img_data'], batch[0]['seg_label']
        else:
            x, y, _ = batch
        x, y = x.float(), y.long()
        
        # CutMix and MixUp
        # x, y = self.cutmix_or_none(x, y)
        
        x_embeddings = self.forward(x, y)
        
        # save predictions to plot tsne
        self.last_X = x_embeddings
        self.last_y = y
        
        start_time = time.time()
        loss_val = self.loss(x_embeddings, y)
        end_time = time.time()
        
        # compute time
        total_time = end_time - start_time

        # save log time
        self.log('total_time', total_time, prog_bar=False, logger=True)
        
        # log loss
        self.log('train_loss', loss_val, prog_bar=True, logger=True)
        
        # self.jaccard_train(x_embeddings, y)
        # self.log('jac_train', self.jaccard_train, prog_bar=True, logger=False)
        
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        
        if torch.isnan(loss_val):
            exit()
        
        return loss_val

    def training_epoch_end(self, outs):
        # self.log('train_jac_epoch', self.jaccard_train, on_step=False, on_epoch=True) 
        return None
        
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x, y = x.float(), y.long()
        x_embeddings = self.forward(x, y)
        
        # save predictions to plot tsne
        self.last_X = x_embeddings
        self.last_y = y
        
        loss_val = self.loss(x_embeddings, y)
        self.log("val_loss", loss_val)
        
        self.jaccard_val(x_embeddings, y)
        self.log('jac_val', self.jaccard_val, prog_bar=False, logger=True)

        return {'loss': loss_val, 'jac_val': self.jaccard_val._forward_cache}
        
    def validation_epoch_end(self, outs):
        # self.log('val_jac_epoch', self.jaccard_val, on_step=False, on_epoch=True) 
        mean_outputs = {}
        for k in outs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outs]).mean()
        print(mean_outputs)
        
        self.log('val_jac_epoch', mean_outputs['jac_val'], on_step=False, on_epoch=True) 

            
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    
    def configure_optimizers(self):
        if self.args.optim == 'adam': 
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.00001)
            return [optimizer]
        elif self.args.optim == 'nadam':
            optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.0001, weight_decay=0.00001)
            # optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.001, weight_decay=0.01)
            return [optimizer]
        elif self.args.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.05)
            return [optimizer]
        elif self.args.optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0001, weight_decay=0.00001)
            return [optimizer]
        else: # sgd 
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.w_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=0.0001)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.t_max, eta_min=0.0001)
            return [optimizer], [scheduler]
    
    def train_dataloader(self):
        if self.train_sampler is None:
            return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.args.num_workers, pin_memory=True, shuffle=False, drop_last=True)
        return DataLoader(self.trainset, sampler=self.train_sampler, batch_size=self.batch_size, num_workers=self.args.num_workers, pin_memory=True, shuffle=False, drop_last=True)
    
    def val_dataloader(self):
        if self.valid_sampler is None:
            return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.args.num_workers, pin_memory=True, shuffle=False, drop_last=True)
        return DataLoader(self.valset, sampler=self.valid_sampler, batch_size=self.batch_size, num_workers=self.args.num_workers, pin_memory=True, shuffle=False, drop_last=True)
    
    # https://junliu-cn.github.io/posts/2020/06/CutMix-Pytorch-Implementation/
    # https://www.kaggle.com/code/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix
    def cutmix(self, data, targets, alpha=1.0):
        # data, targets = batch

        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        lam = np.random.beta(alpha, alpha)

        image_h, image_w = data.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
        targets[:, y0:y1, x0:x1] = shuffled_targets[:, y0:y1, x0:x1]
        # targets = (targets, shuffled_targets, lam)
        
        # d = data[0]
        # t = targets[0]
        
        return data, targets
    
    def cutNone(self, data, targets):
        return data, targets