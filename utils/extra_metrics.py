from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import numpy as np
import os
from argparse import ArgumentParser

# from ood_metrics import auroc, aupr, fpr_at_95_tpr

# OVO x 1xAll
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    print("Computing AUROC to Class.:", per_class)
    
    #creating a list of all the classes except the current class
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

def compute_auc(predictions, labels, categories, file_path):

    # _, _, _, ind_auc, ov_auc = roc_auc_os(predictions, labels)
    # auc = roc_auc_score(labels, predictions, labels=categories, average=None, multi_class='ovr')
    # mauc = roc_auc_score(labels, predictions, labels=categories, average='macro')
    all_auc = roc_auc_score_multiclass(labels, predictions)
    
    # mean
    mauc = np.nanmean(list(all_auc.values()))
    
    # desvio padrão
    std_auc = np.nanstd(list(all_auc.values()))
    
    # include
    all_auc = dict(zip(categories, list(all_auc.values())))
    all_auc['all'] = mauc
    all_auc['std'] = std_auc
   
    print('Computed AUROC.: ' + str(mauc))

    with open(file_path, 'w') as f:
        f.write("%s,%s\n"%('class','auroc'))

    with open(file_path, 'a') as f:
        for key in all_auc.keys():
            f.write("%s,%s\n"%(key, all_auc[key]))

# AUPR por Classe:         
def compute_extra(predictions, labels, categories, file_path):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(labels)
    # unique_class = np.unique(labels)
    aupr_dict = {}
    fpr95_dict = {}
    roc_auc_dict = {}
    
    for per_class in unique_class: 
            
        #creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in labels]
        new_pred_class = [0 if x in other_class else 1 for x in predictions]

        # AUPR por classe
        precision, recall, _ = precision_recall_curve(new_actual_class, new_pred_class)  
        # aupr_dict[i] = np.trapz(precision, recall)
        aupr_dict[per_class] = auc(precision, recall)
        # aupr_dict[per_class] = aupr(new_pred_class, new_actual_class)
        
        # FPR95 por classe
        fpr, tpr, _ = roc_curve(new_actual_class, new_pred_class)
        fpr95_dict[per_class] = tpr[np.argmax(fpr >= 0.95)]
        # fpr95_dict[per_class] = fpr_at_95_tpr(new_pred_class, new_actual_class)
        
        # AUROC por classe
        roc_auc_dict[per_class] = roc_auc_score(new_actual_class, new_pred_class, average="macro")
        # roc_auc_dict[per_class] = auroc(new_pred_class, new_actual_class)
    
    # mean -> include categories
    maupr = np.nanmean(list(aupr_dict.values()))
    all_aupr = dict(zip(categories, list(aupr_dict.values())))
    all_aupr['all'] = maupr
    
    mfpr95 = np.nanmean(list(fpr95_dict.values()))
    all_fpr95 = dict(zip(categories, list(fpr95_dict.values())))
    all_fpr95['all'] = mfpr95
    
    mauroc = np.nanmean(list(roc_auc_dict.values()))
    all_auroc = dict(zip(categories, list(roc_auc_dict.values())))
    all_auroc['all'] = mauroc
    
    # save aupr
    with open(os.path.join(file_path, 'aupr_report.csv'), 'w') as f:
        f.write("%s,%s\n"%('class','aupr'))

    with open(os.path.join(file_path, 'aupr_report.csv'), 'a') as f:
        for key in all_aupr.keys():
            f.write("%s,%s\n"%(key, all_aupr[key]))
            
    # save fpr95
    with open(os.path.join(file_path, 'fpr95_report.csv'), 'w') as f:
        f.write("%s,%s\n"%('class','fpr95'))

    with open(os.path.join(file_path, 'fpr95_report.csv'), 'a') as f:
        for key in all_fpr95.keys():
            f.write("%s,%s\n"%(key, all_fpr95[key]))
            
    # save auroc
    with open(os.path.join(file_path, 'auroc_report.csv'), 'w') as f:
        f.write("%s,%s\n"%('class','auroc'))

    with open(os.path.join(file_path, 'auroc_report.csv'), 'a') as f:
        for key in all_auroc.keys():
            f.write("%s,%s\n"%(key, all_auroc[key]))

if __name__ == '__main__':
    
    parser = ArgumentParser(description="")
    
    parser.add_argument('--folder_id', type=str, default='', help="directory to load args and models")
    # parser.add_argument('--k_idx', type=str, help='') 
    
    args = parser.parse_args()
    
    root_path = '/home/anderson/Documents/large-margin-learning/outputs'

    args.k_idx = 'holdout'
    
    list_report = ['openset_closeset', 'gopenipcs']
    
    for r in list_report:
    
        file_path = os.path.join(root_path, args.folder_id, args.k_idx, 'report', r)
        
        print(file_path)
    
        # load labels and predictions
        y = np.load(os.path.join(file_path, 'labels.npy'))
        y_pred = np.load(os.path.join(file_path, 'predictions.npy'))

        # categorias
        categories = ["background", "anchor", "bird", "butterfly", "cat", "crown", "diamond", "dog", "eagle", "fire", "fish", "flower", "fox", "gun", "heart", "key", "knife", "leaf", "lion", "mermaid", "octopus", "owl", "ribbon", "scorpion", "shark", "shield", "skull", "snake", "spide", "star", "tiger", "water", "wolf", "unknown"]
        
        print('Save ROC AUC score ...')
        compute_auc(y_pred, y, categories, os.path.join(file_path, 'auc_report.csv'))
   
    