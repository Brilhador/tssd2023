from argparse import ArgumentParser

output_dir = "/home/anderson/Documents/large-margin-learning/outputs/"

def parse_args_train_closeset():

    parser = ArgumentParser(description="Close Set Semantic Segmentation")

    # global seed
    parser.add_argument('--seed', type=int, default='1234', help='global seed')

    # model settings
    parser.add_argument('--model', type=str, default='seg_former', choices=['swin', 'seg_former'], help='model name')
    parser.add_argument('--pretrained', type=bool, default=True, help='pre trained on IMAGENET1k_V1')

    # dataset setting
    parser.add_argument('--dataset', type=str.lower, default='tattoo', choices=['tattoo'], help='dataset name')
    parser.add_argument('--custom_classes', type=bool, default=True, help="")
    
    parser.add_argument('--split', type=str.lower, default='train', choices=['train', 'val', 'train+val','test'], help='dataset split step')
    parser.add_argument('--batch_size', type=int, default=16, help='set the batch size')
    
    parser.add_argument('--openset_idx', nargs='*', type=int, default=[], help='class indexes defined as unknown')
    parser.add_argument('--input_width', type=int, default=224, help='input width of model')  # cityscapes
    parser.add_argument('--input_height', type=int, default=224, help='input height of model') # cityscapes
    parser.add_argument('--num_workers', type=int, default=4, help='the number of parallel threads')
    parser.add_argument('--prob_augmentation', type=float, default=0.5, help='probability of apply the data augmentation')

    # parser.add_argument('--mode_aug', type=bool, default=True, help='data augmentation in voc 2012')

    parser.add_argument('--num_images', type=int, default=100, help='num images in dataset geometry')

    # trainnig settings
    parser.add_argument('--train_type', type=str, default='holdout', choices=['holdout'], help='train type: cross-validation or holdout') # only cross implemented
    # parser.add_argument('--k_fold', type=int, default='3', help='number of k-folds')
    # parser.add_argument('--test_size', type=float, default=0.3, help='test size in float (0.2 = 20%)')
    
    parser.add_argument('--max_epochs', type=int, default=5, help='the number of epochs')
    
    parser.add_argument('--optim', type=str.lower,default='nadam',choices=['nadam','sgd','adam','rmsprop', 'adamw'], help="select optimizer")
    parser.add_argument('--lr', type=float, default=0.02, help="initial learning rate")
    parser.add_argument('--w_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--momentum', type=float,default=0.09, help="define momentum value")
    
    # losses settings
    parser.add_argument('--losses', type=str.lower, nargs='+', default=['lm_cross_entropy', 'lm_focal_loss'], choices=['cross_entropy', 'lm_cross_entropy', 'focal_loss', 'lm_focal_loss'], help="select loss functons")
    parser.add_argument('--losses_weight', type=float, nargs='+', default=[0.01, 1.0])

    parser.add_argument('--lam', type=float, default=0.3, help="define the margin global to loss functons")
    parser.add_argument('--magnitude', type=float, default=3, help="")

    # plot TSNE
    parser.add_argument('--plot_tsne', type=bool, default=False, help="Plot latent space 3D")
    
    parser.add_argument('--metrics', type=str.lower, nargs='*', default=['miou'], choices=['fscore', 'miou'], help="select metrics functons")
    
    parser.add_argument('--early_stopping', type=bool, default=False, help='early stopping based in val_loss')   
    parser.add_argument('--patience', type=int, default=10, help="define patience value")

    parser.add_argument('--feature_extractor', type=bool, default=False, help='')   
    
    # cuda settings
    parser.add_argument('--device', type=str, default='cuda', help="running on CPU or CUDA (GPU)")
    parser.add_argument('--visible_device', type=str, default='1', help="select visible device")
    
    # local save
    parser.add_argument('--savedir', default=output_dir, help="directory to save the model snapshot")

    args = parser.parse_args()
    
    return args

def parse_args_train_report():

    parser = ArgumentParser(description="Close Set Semantic Segmentation - Report")

    # global seed
    parser.add_argument('--seed', type=int, default='1234', help='global seed')

    parser.add_argument('--folder_id', type=str, default='', help="directory to load args and models")
    parser.add_argument('--folder_path', type=str, default=output_dir, help="directory to load args and models")     
    parser.add_argument('--save_images', type=bool, default=True, help="save images from predictions") 
    parser.add_argument('--num_save_images', type=int, default=500, help="save images from predictions") 
    parser.add_argument('--posprocessing', type=str, default='None', choices=['slic', 'opening', 'None'], help="")
    
    parser.add_argument('--k_idx', type=str, help='') 
    parser.add_argument('--name_checkpoint', type=str, help='') 
     
    args = parser.parse_args()
    
    return args, parser

def parse_args_openset_openipcs():
    
    parser = ArgumentParser(description="Open Set Semantic Segmentation - Report")
    
    # global seed
    parser.add_argument('--seed', type=int, default='1234', help='global seed')

    # IPCS settings
    parser.add_argument('--n_components', type=int, default=64, help="")
    
    parser.add_argument('--folder_id', type=str, default='', help="directory to load args and models")
    parser.add_argument('--folder_path', type=str, default=output_dir, help="directory to load args and models")    
    parser.add_argument('--save_images', type=bool, default=True, help="save images from predictions")  
    parser.add_argument('--num_save_images', type=int, default=500, help="save images from predictions") 
    parser.add_argument('--posprocessing', type=str, default='None', choices=['slic', 'opening', 'None'], help="")
    
    parser.add_argument('--k_idx', type=str, help='') 
    parser.add_argument('--name_checkpoint', type=str, help='') 
    
    # only use geometry dataset
    parser.add_argument('--num_images', type=int, default=100, help='num images in dataset geometry')  

    # metric eval
    parser.add_argument('--metric_eval', type=str, default='miou', choices=['f1score', 'miou'], help="metric to eval best search")

    args = parser.parse_args()
    
    return args, parser