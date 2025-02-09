import os
from PIL import Image
import json

# /mnt/data/Doutorado/dataset/tattoo/annotations
# /mnt/data/Doutorado/dataset/tattoo/images

def create_odgt(root, out_dir):
    
    # get dir images and annotations
    images_dir = os.path.join(root, 'images')
    targets_dir = os.path.join(root, 'mask_ids')
    
    # list of dict to dump in file
    out_dir_files = []
    
    # get all images and targets
    list_target = sorted(os.listdir(targets_dir))
    list_image = sorted(os.listdir(images_dir))
        
    for file_target, file_image in zip(list_target, list_image):

        target_ = os.path.join(targets_dir, file_target)
        image_ = os.path.join(images_dir, file_image)
        
        image = Image.open(image_)
        width, height = image.size
        
        dict_entry = {
            "dbName": "TattooSemanticSegmentation",
            "width": width,
            "height": height,
            "fpath_img": image_,
            "fpath_segm": target_,
        }

        out_dir_files.append(dict_entry)
            
    print("Total images: " + str(len(out_dir_files)))
            
    with open(out_dir, "w") as outfile:
        json.dump(out_dir_files, outfile)
            
def main():
    # out_dir = "tattoo_train.odgt"
    # root_dir = "/mnt/code/Doutorado/code/segmentation/semantic/prototypical-triplet-tattoo-openset-segmentation/dataset/tattoo/train"
    # out_dir = os.path.join(root_dir, out_dir)
    # # out_dir = os.path.join("/mnt/code/Doutorado/code/segmentation/semantic/pytorch-background-shift-segmentation/datasets", out_dir)
    # create_odgt(root_dir, out_dir)
    
    # out_dir = "tattoo_val.odgt"
    # root_dir = "/mnt/code/Doutorado/code/segmentation/semantic/prototypical-triplet-tattoo-openset-segmentation/dataset/tattoo/val"
    # out_dir = os.path.join(root_dir, out_dir)
    # # out_dir = os.path.join("/mnt/code/Doutorado/code/segmentation/semantic/pytorch-background-shift-segmentation/datasets", out_dir)
    # create_odgt(root_dir, out_dir)
    
    out_dir = "tattoo_test_closed.odgt"
    root_dir = "/mnt/code/Doutorado/code/segmentation/semantic/prototypical-triplet-tattoo-openset-segmentation/dataset/tattoo/test_closed"
    out_dir = os.path.join(root_dir, out_dir)
    # out_dir = os.path.join("/mnt/code/Doutorado/code/segmentation/semantic/pytorch-background-shift-segmentation/datasets", out_dir)
    create_odgt(root_dir, out_dir)
    
    out_dir = "tattoo_test_open.odgt"
    root_dir = "/mnt/code/Doutorado/code/segmentation/semantic/prototypical-triplet-tattoo-openset-segmentation/dataset/tattoo/test_open"
    out_dir = os.path.join(root_dir, out_dir)
    # out_dir = os.path.join("/mnt/code/Doutorado/code/segmentation/semantic/pytorch-background-shift-segmentation/datasets", out_dir)
    create_odgt(root_dir, out_dir)
    
if __name__ == '__main__':
    main()