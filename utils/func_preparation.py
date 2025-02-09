# ignore deprecations warning
import warnings
warnings.filterwarnings("ignore")

import os
import errno
import time
import itertools
import random
import numpy as np
import pandas as pd
import cv2
from skimage import util

import torch
from torch import optim
from torch import nn
from utils import losses
# from utils import metrics

import matplotlib.pyplot as plt

from fastdist import fastdist
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

# from .iou import IoU
from torchmetrics.classification import MulticlassJaccardIndex

from sklearn.metrics import confusion_matrix

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from .CustomProgressBar import CustomRichProgressBar
from .trackers import MetricTracker
            
from dataset.base_dataset import BaseDataset

# plot T-SNE
# from tsne_torch import TorchTSNE as TSNE
from sklearn.manifold import TSNE

from skimage.segmentation import slic, watershed
# from cuda_slic import slic

from models.seg_module_base import SegmentationModuleBase
           
# transformers
from transformers import SwinConfig, UperNetConfig # Swin 
from models.modeling_segformer import SegformerConfig, SegformerForSemanticSegmentation # SegFormer
from models.modeling_upernet import UperNetForSemanticSegmentation # Swin            
            
def get_model(args, feature_extraction=False):
    
    if args.model == 'seg_former':
        if args.pretrained:
            model =  SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b5", 
                return_dict=False, 
                num_labels=args.num_classes,
                ignore_mismatched_sizes=True,
            ) # Fine-tune
        else:
            configuration = SegformerConfig()
            configuration.num_labels = args.num_classes
            configuration.return_dict = False
            configuration.ignore_mismatched_sizes = True
            # b1
            # configuration.hidden_sizes = [64, 128, 320, 512]
            # configuration.decoder_hidden_size = 256
            # configuration.depths = [2, 2, 2, 2]
            # b2
            # configuration.hidden_sizes = [64, 128, 320, 512]
            # configuration.decoder_hidden_size = 768
            # configuration.depths = [3, 4, 6, 3]
            # b3
            # configuration.hidden_sizes = [64, 128, 320, 512]
            # configuration.decoder_hidden_size = 768
            # configuration.depths = [3, 4, 18, 3]
            # # b4
            # configuration.hidden_sizes = [64, 128, 320, 512]
            # configuration.decoder_hidden_size = 768
            # configuration.depths = [3, 8, 27, 3]
            # b5
            configuration.hidden_sizes = [64, 128, 320, 512]
            configuration.decoder_hidden_size = 768
            configuration.depths = [3, 6, 40, 3]
            model = SegformerForSemanticSegmentation(configuration)
    elif args.model == 'swin':
        if args.pretrained:
            model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-base")
            # model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")
            # model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-small")
            config = model.config
            config.num_labels = args.num_classes
            config.return_dict = False
            model.decode_head.classifier = nn.Conv2d(512, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        else: 
            backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
            config = UperNetConfig(backbone_config=backbone_config)
            config.num_labels = args.num_classes
            config.return_dict = False
            config.torch_dtype = "float32"
            config.layer_scale_init_value = 1e-06
            config.architectures = ["ConvNextV2+UperNet"]
            model = UperNetForSemanticSegmentation(config)
    else: 
        exit()
    
    return model

def load_model(args, feature_extraction=False):
    
    checkpoints_path = os.path.join(args.savedir, str(args.k_idx), 'checkpoints')
    checkpoints_path = os.path.join(checkpoints_path, args.name_checkpoint)
    print(checkpoints_path)
    
    # update parans to train 
    model = get_model(args)    
    loss = get_loss(args)
    
    model = SegmentationModuleBase.load_from_checkpoint(
            checkpoints_path,
            args=args,
            model=model,
            loss=loss,
            batch_size=args.batch_size,
            trainset=None,
            train_sampler=None,
            valset=None,
            valid_sampler=None,
            num_classes=args.num_classes
        )

    model.cuda()
    
    # define extract feature
    if feature_extraction:
        if args.model == 'seg_former' or args.model == 'swin':
            model.model.config.output_hidden_states = True
            model.model.config.output_attentions = True
          
    model.model.eval()
    
    return model

def get_dataset_train(args, train_transforms, valid_transforms):
    
    root_odgt = '/mnt/code/Doutorado/code/segmentation/semantic/pytorch-background-shift-segmentation/datasets'
    
    if args.dataset == 'tattoo':  
        file_train_odgt = 'tattoo_train.odgt'
        file_val_odgt = 'tattoo_val.odgt'
    else:
        raise NotImplementedError('This dataset not supported')

    odgt_train = os.path.join(root_odgt, file_train_odgt)
    odgt_val = os.path.join(root_odgt, file_val_odgt)
    
    train_dataset = BaseDataset(args=args, odgt=odgt_train, augmentation=train_transforms(args))
    valid_dataset = BaseDataset(args=args, odgt=odgt_val, augmentation=valid_transforms(args))
        
    return train_dataset, valid_dataset

def get_dataset_test(args, train_transforms, valid_transforms):
    
    root_odgt = '/mnt/code/Doutorado/code/segmentation/semantic/pytorch-background-shift-segmentation/datasets'
    
    if args.dataset == 'tattoo':  
        if args.split == 'train':
            file_odgt = 'tattoo_test_closed.odgt'
        else:
            file_odgt = 'tattoo_test_open.odgt'
    else:
        raise NotImplementedError('This dataset not supported')
     
    odgt_train = os.path.join(root_odgt, file_odgt)
    
    train_dataset = BaseDataset(args=args, odgt=odgt_train, augmentation=train_transforms(args))
    valid_dataset = BaseDataset(args=args, odgt=odgt_train, augmentation=valid_transforms(args))
        
    return train_dataset, valid_dataset

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_loss(args):
    
    list_losses = []
    
    for loss in args.losses:
    
        if 'cross_entropy' == loss:
            list_losses.append(losses.CrossEntropyLoss(ignore_index=255))
               
        if 'focal_loss' == loss:
            list_losses.append(losses.FocalLoss(ignore_index=255))
        
        if 'lm_cross_entropy' == loss:
            list_losses.append(losses.LargeMarginInSoftmaxLoss(lam=args.lam, ignore_index=255))
            
        if 'lm_focal_loss' == loss:
            list_losses.append(losses.LargeMarginFocalLoss(lam=args.lam, ignore_index=255))
            
    loss = losses.CombinedLoss(losses=list_losses, weights=args.losses_weight)
    
    return loss

def get_optimizer(args, model):
    
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    else:
        raise NotImplementedError('This network not supported')  
    
    return optimizer

def get_callbacks(args):
    
    progressbar = CustomRichProgressBar()
    early_stop_callback = EarlyStopping(
            monitor="val_jac_epoch", 
            mode="max", 
            patience=args.patience
        )
    
    if np.array(args.openset_idx).shape[0] == 0:
        name = args.loss + '_' + args.dataset + '_closeset_'
    else: 
        idx = '_'.join(map(str, args.openset_idx))
        name = args.loss + '_' + args.dataset + '_' + 'unknown_class_' + idx + '_'
        
    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(args.savedir, 'checkpoints'),
        monitor = 'val_jac_epoch',
        mode = 'max',
        verbose = True,
        filename= name + '{epoch:02d}-{val_jac_epoch:.3f}'
    )
    
    tracker = MetricTracker(args)
    
    return [progressbar, early_stop_callback, checkpoint_callback, tracker]

# https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
def get_distance(arr1, arr2, distance_type):

    dist = 0.0
    
    if distance_type == 'euclidean':
        dist = np.sqrt((np.square(arr1 - arr2).sum(axis=1)))
    else:
        print("distance type not known: use euclidean")
    
    return dist

def compute_distance(mavs, logits, distance_type):

    dists = []

    for l in logits:

        if distance_type == 'euclidean':
            dists.append([np.sqrt((np.square(l - mvc).sum())) for mvc in mavs])
        else:
            print("distance type not known: enter either of euclidean or dtw")

    return np.array(dists)  

def prepare_data_search(num_samples, num_classes, labels, outs):
    
    # count number labels per input
    count = [0 for _ in range(num_classes)] # list of list

    outs_scores = []
    outs_labels = []

    # labels
    labels = torch.squeeze(labels.reshape((labels.size()[0] * labels.size()[1] * labels.size()[2])))
    labels = labels.detach().cpu().numpy()

    # logits
    outs = outs.permute(0, 2, 3, 1)
    outs = torch.reshape(outs, (outs.size()[0] * outs.size()[1] * outs.size()[2], outs.size()[3]))
    outs = outs.detach().cpu().numpy()

    # remove ignore index
    mask = np.ones(len(labels), dtype=bool)
    mask[labels == 255] = False
    
    labels = labels[mask,...]
    outs = outs[mask,...]
    
    # shuffle value
    mix = list(zip(labels, outs))
    random.shuffle(mix)
    labels, outs = zip(*mix)
    
    # scroll through the pixels
    for l, o in zip(labels, outs):
        # print(l)
        if count[l] < num_samples:
            outs_scores.append(o)
            outs_labels.append(l)
            count[l] += 1

    return np.array(outs_labels), np.array(outs_scores)

def prepare_data(labels, outs):
    
    # labels
    labels = torch.squeeze(labels.reshape((labels.size()[0] * labels.size()[1] * labels.size()[2])))
    labels = labels.detach().cpu().numpy()

    # logits
    outs = outs.permute(0, 2, 3, 1)
    outs = torch.reshape(outs, (outs.size()[0] * outs.size()[1] * outs.size()[2], outs.size()[3]))
    outs = outs.detach().cpu().numpy()

    return np.array(labels), np.array(outs)

def init_preds(args):

    pred_y, pred_y_o, all_labels = {}, {}, {}

    for tailsize in args.tailsize:

        pred_y[tailsize] = {}
        pred_y_o[tailsize] = {}
        all_labels[tailsize] = {}

        for alpha in args.alpha:

            pred_y[tailsize][alpha] = {}
            pred_y_o[tailsize][alpha] = {}
            all_labels[tailsize][alpha] = {}

            for th in args.th:

                pred_y[tailsize][alpha][th] = []
                pred_y_o[tailsize][alpha][th] = []
                all_labels[tailsize][alpha][th] = []

    return pred_y, pred_y_o, all_labels 

def init_pred_ood(args):    
    pred = {}
    for th in args.th:
        pred[th] = []
        
    return pred
    
def init_history(args):
    
    history = [[] for _ in range(args.k_fold)]
    
    for k in range(args.k_fold):
        
        history[k] = {}
        
        history[k]['num_epochs'] = args.max_epochs
        
        history[k]['train_loss'] = {}
        history[k]['train_loss'][args.loss] = []
        
        history[k]['valid_loss'] = {}
        history[k]['valid_loss'][args.loss] = []
        
        history[k]['train_metric'] = {}
        history[k]['valid_metric'] = {}
        
        for metric in args.metrics:
            history[k]['train_metric'][metric] = []
            history[k]['valid_metric'][metric] = []
            
    return history

def save_plot_history(args, history):
    
    file_path = os.path.join(args.savedir, "history")
    create_dir(file_path)
    
    for k in range(0, args.k_fold): 
        
        plt.title("Plot Train-Val Loss K = " + str(k+1))
        plt.plot(range(1, history[k]['num_epochs']+1), history[k]['train_loss'][args.loss], label="train_loss")
        plt.plot(range(1, history[k]['num_epochs']+1), history[k]['valid_loss'][args.loss], label="valid_loss")
        plt.ylabel(args.loss)
        plt.xlabel("Training Epochs")
        plt.legend()
        
        file_name = "loss_plot_k_" + str(k+1) + ".png"
        plt.savefig(os.path.join(file_path, file_name))
        
        # clear
        plt.cla()  
        
    for k in range(0, args.k_fold): 
        
        if len(args.metrics) > 0:
        
            plt.title("Plot Train-Val Metrics K = " + str(k+1))
            
            for metric in args.metrics:
                plt.plot(range(1, history[k]['num_epochs']+1), history[k]['train_metric'][metric], label="train_" + metric)
                plt.plot(range(1, history[k]['num_epochs']+1), history[k]['valid_metric'][metric], label="valid_" + metric)
            
            plt.ylabel("Metrics")
            plt.xlabel("Training Epochs")
            plt.legend()
            
            file_name = "metrics_plot_k_" + str(k+1) + ".png"
            plt.savefig(os.path.join(file_path, file_name))
            
            # clear
            plt.cla()

def early_stopping_loss(args, best_overall, best_score, val_loss, count_patience, early_stop, model, k):
    
    if best_score > val_loss:
        best_score = val_loss
        if best_overall > best_score:
            print(f'Validation loss decreased ({best_overall:.6f} --> {val_loss:.6f}).  Saving best overall model ...')
            best_overall = best_score
            # save model
            save_model(args, model)
            save_best_model_history(args, best_overall, k)
        # resetting value
        count_patience = 0
    else:
        count_patience += 1
        if count_patience >= args.patience:
            early_stop = True
            
    return best_overall, best_score, count_patience, early_stop

def early_stopping_metric(args, best_overall, best_score, val_metric, count_patience, early_stop, model, k):
    
    if best_score < val_metric:
        best_score = val_metric
        if best_overall < best_score:
            print(f'Validation metric increased ({best_overall:.6f} --> {val_metric:.6f}).  Saving best overall model ...')
            best_overall = best_score
            # save model
            save_model(args, model)
            save_best_model_history(args, best_overall, k)
        # resetting value
        count_patience = 0
    else:
        count_patience += 1
        if count_patience >= args.patience:
            early_stop = True
            
    return best_overall, best_score, count_patience, early_stop

def save_model(args, model):
    
    file_path = os.path.join(args.savedir, 'best_overall_model.pth')
    torch.save(model, file_path)
    
def save_best_model_history(args, val_loss, k):
    
    file_path = os.path.join(args.savedir, 'best_overall_history.txt')
    f = open(file_path, 'a')
    f.write('K-fold.: ' + str(k+1) + " -- " + \
            'Val_loss.: ' + str(val_loss) + "\n")
    f.close()

def make_outputs_dir(args):
    idx = '_'.join(map(str, args.openset_idx))
    # if args.loss is None:
    args.loss = ''
    for l in args.losses:
        args.loss = args.loss + l + '_'
        
    name_folder = args.model + '_' + args.loss + '_' + args.dataset + '_' + 'unknown_class_' + idx + '_' + str(int(round(time.time() * args.seed)))
    args.savedir = os.path.join(args.savedir, name_folder)
    create_dir(args.savedir)

def reproducibility(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
 
def create_dir(path):
    ## criando o diretorio
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # else:
    #     print("Directory already exists: " + path)
        
def segmap_to_rgb(input, num_class, class_index, colors=None):

    if colors is None: 
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor(class_index)[:, None] * palette
        colors = torch.as_tensor([i for i in range(num_class)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

    r = np.zeros_like(input).astype(np.uint8)
    g = np.zeros_like(input).astype(np.uint8)
    b = np.zeros_like(input).astype(np.uint8)

    for i, c in enumerate(class_index):
    # for c in class_index:
        idx = input == c
        r[idx] = colors[i, 0]
        g[idx] = colors[i, 1]
        b[idx] = colors[i, 2]

    rgb = np.stack([r, g, b], axis=2)

    return rgb

def compute_iou(predictions, labels, categories, file_path, ignore_index=None):
        
    iou = jaccard_score(labels, predictions, average=None)
    miou = np.nanmean(iou)
    
    all_iou = dict(zip(categories, iou))
    all_iou['all'] = miou
    
    with open(file_path, 'w') as f:
        f.write("%s,%s\n"%('class','miou'))

    with open(file_path, 'a') as f:
        for key in all_iou.keys():
            f.write("%s,%s\n"%(key, all_iou[key]))

def compute_iou_value(predictions, labels, num_classes, ignore_index=None):
    
    iou = jaccard_score(labels, predictions, average=None)
    miou = np.nanmean(iou)
    
    return miou

def save_plot_confusion_matrix(predictions, labels, categories, file_path):

    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    plot_confusion_matrix(cm=cm, target_names=categories, file_path=file_path, title='Closet Set Semantic Segmentation')

def plot_confusion_matrix(cm, target_names, file_path, title='Confusion matrix', cmap=None, normalize=True):

    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        # cmap = plt.get_cmap('RdYlGn')

    plt.figure(figsize=(35, 35))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.01 if normalize else cm.max() / 2
    # print(cm.max())
    # print(thresh)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()

    # save image
    plt.savefig(file_path)

def generete_report(args, y_pred, y, y_names, dir_name, y_score=None):
    
    # y_pred, y, y_names = compute_predictions_closetset(args)
    
    file_path = os.path.join(args.savedir, str(args.k_idx), "report", dir_name)
    create_dir(file_path)

    # remove pixels from ignored index 
    mask = np.ones(len(y), dtype=bool)
    mask[y == 255] = False
    y = y[mask,...]
    y_pred = y_pred[mask,...]
    
    if y_score is not None:
        y_score = y_score[mask,...]
        np.save(os.path.join(file_path, 'embeddings.npy'), y_score)
    
    # report
    print('Generate classification report ...')
    report = classification_report(y, y_pred, target_names=y_names, output_dict=True, digits=6)
    report = pd.DataFrame(report).transpose()
    
    # save report
    print('Save classification report ...')
    report.to_csv(os.path.join(file_path, 'classification_report.csv'), index= True)
    
    # matrix confusion
    print('Save plot confunsion matrix ...')
    save_plot_confusion_matrix(y_pred, y, y_names, os.path.join(file_path, 'confusion_matrix.jpg'))

    # mean IOU per class
    print('Save mean iou (with background) per class report ...')
    compute_iou(y_pred, y, y_names, os.path.join(file_path, 'miou_report.csv'))
    
    # overral accuracy
    print('Save overrall accuracy ...')
    oa = accuracy_score(y, y_pred)
    np.savetxt(os.path.join(file_path, 'overrall_accuracy.txt'), [oa], fmt='%0.4f')
    
    # Kappa score
    print('Save Kappa score ...')
    ka = cohen_kappa_score(y, y_pred, labels=None, weights=None)
    np.savetxt(os.path.join(file_path, 'kappa_score.txt'), [ka], fmt='%0.4f')
    
    # ROC AUC SCORE
    print('Save roc auc score ...')
    
    # Save predictions and labels
    print('Save predictions and labels ...')
    np.save(os.path.join(file_path, 'predictions.npy'), y_pred)
    np.save(os.path.join(file_path, 'labels.npy'), y)
        
    # Acc(k) - Acurácia binária entre KKCs and UUCs (não faz sevntido utilizar deido as desbalanceamento das classes)
    # Pre(u) - Precisão dos pixels desconhecidos - (F1-Score)
    # Kappa(k) - Avalia o desempenho geral da segmentação (comum no sensoriamento remoto)
    
    # # matrix confusion
    # mask = np.ones(len(y), dtype=bool)
    # mask[y == 0] = False
    # y = y[mask,...]
    # y_pred = y_pred[mask,...]
    # y_names = y_names[1:]
    
    # print(y_names)
    
    # print('Save plot confunsion matrix without background ...')
    # save_plot_confusion_matrix(y_pred, y, y_names, os.path.join(file_path, 'confusion_matrix_with_background.jpg'))

def generete_report_unknown(args, y_pred, y, dir_name):
    
    file_path = os.path.join(args.savedir, str(args.k_idx), "report", dir_name)
    create_dir(file_path)

    # remove pixels from ignored index 
    mask = np.ones(len(y), dtype=bool)
    mask[y == 255] = False
    y = y[mask,...]
    y_pred = y_pred[mask,...]
    
    # binary mIoU (known and unknown)
    pred_binary = np.zeros(len(y), dtype=int)
    y_binary = np.zeros(len(y), dtype=int)
    unknown_id = np.max(np.unique(y))
    pred_binary[y_pred == unknown_id] = 1
    y_binary[y == unknown_id] = 1
    
    # mean IOU per class
    print('Save mean iou (with background) per class report ...')
    compute_iou(pred_binary, y_binary, ['known', 'unknown'], os.path.join(file_path, 'miou_report_binary.csv'))

def save_best_scores_openipcs(args, best_score, th_score_best, metric_eval):
    
    dir_path = os.path.join(args.savedir, str(args.k_idx), 'report', 'gopenipcs')
    create_dir(dir_path)
     
    file_name = os.path.join(dir_path, 'best_params_openipcs.txt')
    f = open(file_name, 'a')
    f.write(str(best_score))
    f.write("\n" + str(th_score_best))
    f.write("\n" + metric_eval)
    f.close()

def save_openipcs_predictions(args, image, label, pred, classes_idx, image_name, colors=None):
    
    # save predictions
    file_path = os.path.join(args.savedir, str(args.k_idx), 'report', 'gopenipcs', 'predictions')
    create_dir(file_path)
    
    plt.figure(figsize=(10,10))

    plt.imshow(image, interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_img_.png').replace('.jpg', '_img_.png').replace('.tif', '_img_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.imshow(segmap_to_rgb(label, len(classes_idx), classes_idx, colors), interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_msk_.png').replace('.jpg', '_msk_.png').replace('.tif', '_msk_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.imshow(segmap_to_rgb(pred, len(classes_idx), classes_idx, colors), interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_gopenipcs_.png').replace('.jpg', '_gopenipcs_.png').replace('.tif', '_gopenipcs_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.close('all')
    
def save_closeset_predictions(args, image, label, pred, classes_idx, image_name, colors=None):
    
    # save predictions
    file_path = os.path.join(args.savedir, str(args.k_idx), 'report', 'closeset', 'predictions')
    create_dir(file_path)
    
    plt.figure(figsize=(10,10))

    plt.imshow(image, interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_img_.png').replace('.jpg', '_img_.png').replace('.tif', '_img_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.imshow(segmap_to_rgb(label, len(classes_idx), classes_idx, colors), interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_msk_.png').replace('.jpg', '_msk_.png').replace('.tif', '_msk_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.imshow(segmap_to_rgb(pred, len(classes_idx), classes_idx, colors), interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_closeset_.png').replace('.jpg', '_closeset_.png').replace('.tif', '_closeset_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.close('all')

def save_openset_closeset_predictions(args, image, label, pred, classes_idx, image_name, colors=None):
    
    # save predictions
    # SLIC
    if args.posprocessing == 'slic':
        file_path = os.path.join(args.savedir, str(args.k_idx), 'report', 'openset_closeset_slic', 'predictions')
    # Open Morphological Operation
    elif args.posprocessing == 'opening':
        file_path = os.path.join(args.savedir, str(args.k_idx), 'report', 'openset_closeset_opening', 'predictions')
    else:
        file_path = os.path.join(args.savedir, str(args.k_idx), 'report', 'openset_closeset', 'predictions')
    create_dir(file_path)
    
    plt.figure(figsize=(10,10))
    
    plt.imshow(image, interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_img_.png').replace('.jpg', '_img_.png').replace('.tif', '_img_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.imshow(segmap_to_rgb(label, len(classes_idx), classes_idx, colors), interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_msk_.png').replace('.jpg', '_msk_.png').replace('.tif', '_msk_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.imshow(segmap_to_rgb(pred, len(classes_idx), classes_idx, colors), interpolation="nearest")
    plt.axis('off')
    plt.savefig(os.path.join(file_path , image_name.replace('.png', '_closeset_.png').replace('.jpg', '_closeset_.png').replace('.tif', '_closeset_.png')), bbox_inches='tight', pad_inches = 0)
    
    plt.close('all')

def plot_embeddings3D(args, X, y, categories):

    colors = {0:'red', 1:'blue', 2:'green'}

    print(X.shape)

    # remove ignore index
    mask = np.ones(len(y), dtype=bool)
    mask[y == 255] = False
    X = X[mask,...]
    y = y[mask,...]
    
    # random get indx
    indices = np.random.randint(X.shape[0], size=10000)
    X = X[indices]
    y = y[indices]
    
    # get axes embeddings
    X = X.T
    Xax = X[0]
    Yax = X[1]
    Zax = X[2]
    
    # dict
    cdict = {0:'black',1:'green',2:'blue'}
    # labl = {0:'Malignant',1:'Benign',2:'test'}
    labl = dict(zip(range(3), categories))
    print(labl)
    marker = {0:'*',1:'o',2:'^'}
    alpha = {0:.3, 1:.5, 2:.7}
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection="3d")
    # ax.force_zorder = True
    
    fig.patch.set_facecolor('white')
    for l in np.unique(y):
        ix=np.where(y==l)
        # ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
        #     label=labl[l], marker=marker[l], alpha=alpha[l])
        ax.plot(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], markersize=3,
            label=labl[l], marker=marker[l], alpha=alpha[l], linewidth=0)
    
    # # make simple, plot anchors
    if args.loss == 'prototycal_triplet':
        ax.plot(3, 0, 0, c='r', marker=marker[0], markersize=8)
        ax.plot(0, 3, 0, c='r', marker=marker[1], markersize=8)
        ax.plot(0, 0, 3, c='r', marker=marker[2], markersize=8)
    else:
        ax.plot(0, 0, 0, c='r', marker=marker[0], markersize=8)
        ax.plot(0, 3, 0, c='r', marker=marker[1], markersize=8)
        ax.plot(0, 0, 3, c='r', marker=marker[2], markersize=8)
    
    # for loop ends
    # ax.set_xlabel("First Principal Component", fontsize=14)
    # ax.set_ylabel("Second Principal Component", fontsize=14)
    # ax.set_zlabel("Third Principal Component", fontsize=14)
    legend = ax.legend(loc='best', markerscale=4, fontsize=14)
    
    plt.gca().view_init(20, 45)
    plt.show()
    # file_path = os.path.join(args.savedir, 'embedding_3d.png')
    # plt.savefig(file_path)

def plot_tsne(args, X, y):

    # tsne = TSNE(n_components = 2)
    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, verbose=True) 
    X_hat = tsne.fit_transform(X) # returns shape (n_samples, 2)
    
    # To plot the embedding
    # names = ['class_1', 'class_2', 'class_3', 'class_4']
    for i in range(np.unique(y)):
        X_label = X_hat[np.where(y == i)]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=i)
    plt.legend()
    # plt.show()
    file_path = os.path.join(args.save_dir, 't-sne_perplexity50_embedding.png')
    plt.savefig(file_path)

def plot_tsne_epoch(args, idx_epoch, X, y):

    # tsne = TSNE(n_components = 2)
    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, verbose=True) 
    X_hat = tsne.fit_transform(X) # returns shape (n_samples, 2)
    
    # To plot the embedding
    # names = ['class_1', 'class_2', 'class_3', 'class_4']
    for i in range(np.unique(y)):
        X_label = X_hat[np.where(y == i)]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=i)
    plt.legend()
    # plt.show()
    file_path = os.path.join(args.save_dir, 't-sne_epoch_'+str(idx_epoch)+'.png')
    plt.savefig(file_path)

# https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/footprints.py    
# connectivity = 1
# 0 1 0
# 1 1 1
# 0 1 0
def _cross(r0, r1, dtype=np.uint8):
    """Cross-shaped structuring element of shape (r0, r1).
    Only the central row and column are ones.
    """
    s0 = int(2 * r0 + 1)
    s1 = int(2 * r1 + 1)
    c = np.zeros((s0, s1), dtype=dtype)
    if r1 != 0:
        c[r0, :] = 1
    if r0 != 0:
        c[:, r1] = 1
    return c



def plot_embeddings3D(args, X, y, categories):

    X = X.permute(0, 2, 3, 1).contiguous()
    shape = X.size()
    X = X.view(shape[0] * shape[1] * shape[2], shape[3])
    X = X.detach().cpu().numpy()
    
    y = y.view(shape[0] * shape[1] * shape[2])
    y = y.detach().cpu().numpy()

    # print(X.shape)

    # remove ignore index
    mask = np.ones(len(y), dtype=bool)
    mask[y == 255] = False
    X = X[mask,...]
    y = y[mask,...]
    
    # random get indx
    # indices = np.random.randint(X.shape[0], size=10000)
    # X = X[indices]
    # y = y[indices]
    X, y = select_n_emb_per_class(X, y, 5000)
    
    # get axes embeddings
    X = X.T
    Xax = X[0]
    Yax = X[1]
    Zax = X[2]
    
    # dict
    cdict = {0:'#333333', 1:'#0343ff', 2:'#15b01a', 3:'#FFB833'}
    labl = dict(zip(range(4), categories))
    # print(labl)
    marker = {0:'*',1:'s',2:'o',3:'+'}
    alpha = {0:.3, 1:.5, 2:.5, 3:0.5}
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection="3d")
    # ax.force_zorder = True
    
    fig.patch.set_facecolor('white')
    for l in np.unique(y):
        ix=np.where(y==l)
        # ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
        #     label=labl[l], marker=marker[l], alpha=alpha[l])
        ax.plot(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], markersize=3,
            label=labl[l], marker=marker[l], alpha=alpha[l], linewidth=0)
    
    # # make simple, plot anchors
    if args.loss == 'prototycal_triplet' or args.loss == 'prototycal_random':
        ax.plot(3, 0, 0, c='r', marker=marker[0], markersize=8)
        ax.plot(0, 3, 0, c='r', marker=marker[1], markersize=8)
        ax.plot(0, 0, 3, c='r', marker=marker[2], markersize=8)
    
    # for loop ends
    # ax.set_xlabel("First Principal Component", fontsize=14)
    # ax.set_ylabel("Second Principal Component", fontsize=14)
    # ax.set_zlabel("Third Principal Component", fontsize=14)
    # legend = ax.legend(loc='best', markerscale=4, fontsize=14)
    
    plt.gca().view_init(20, 45)
    # plt.show()
    
    file_path = os.path.join(args.save_plots,'plot_embeddings_3d.png')
    plt.savefig(file_path)
    
def plot_embeddings_2D(args, X, y, categories):

    # resize torch tensor
    # torch to numpy
    X = X.permute(0, 2, 3, 1).contiguous()
    shape = X.size()
    X = X.view(shape[0] * shape[1] * shape[2], shape[3])
    X = X.detach().cpu().numpy()
    
    y = y.view(shape[0] * shape[1] * shape[2])
    y = y.detach().cpu().numpy()

    # print(X.shape)

    # remove ignore index
    mask = np.ones(len(y), dtype=bool)
    mask[y == 255] = False
    X = X[mask,...]
    y = y[mask,...]
    
    # random get indx
    X, y = select_n_emb_per_class(X, y, 5000)
    
    # compute the TSNE
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', n_iter=1000, perplexity=50, verbose=0)
    X_hat = tsne.fit_transform(X) # returns shape (n_samples, 2)
    
    # dict
    cdict = {0:'#333333', 1:'#0343ff', 2:'#15b01a', 3:'#FFB833'}
    labl = dict(zip(range(4), categories))
    # print(labl)
    marker = {0:'*',1:'s',2:'o',3:'+'}
    alpha = {0:.3, 1:.5, 2:.5, 3:0.5}
    
    fig = plt.figure(figsize=(7,5))
    fig.patch.set_facecolor('white')
    
    for i in np.unique(y):
        mask = np.where(y == i)
        X_label = X_hat[mask]
        # plt.scatter(X_label[:, 0], X_label[:, 1], label=i)
        plt.scatter(X_label[:, 0], X_label[:, 1], c=cdict[i], s=40,
            label=labl[i], marker=marker[i], alpha=alpha[i])
        
    # plt.legend()
    # plt.legend(loc='best', markerscale=2, fontsize=10)
    
    # plt.show()
    file_path = os.path.join(args.save_plots,'tsne_plot_embeddings_2d.png')
    plt.savefig(file_path)
    
def select_n_emb_per_class(X, y, n):
        
        X_temp, y_temp = [], []
        for c in np.unique(y):
            tmp_x = X[y == c]
            tmp_y = y[y == c]
            indices = np.random.randint(tmp_x.shape[0], size=n)
            tmp_x = tmp_x[indices]
            tmp_y = tmp_y[indices]
            X_temp.append(tmp_x)
            y_temp.append(tmp_y)
        
        X_temp = np.concatenate(X_temp)
        y_temp = np.concatenate(y_temp)
        
        return X_temp, y_temp
        