import glob

import numpy as np
import torch
import os
import nibabel as nib


def get_prev_training_data(checkpoint_dir):
    last_epoch = 0
    check_points = sorted(glob.glob(checkpoint_dir))
    for check_point in check_points:
        check_epoch = int(check_point.split(os.sep)[-1].split('_')[2])
        if check_epoch > last_epoch:
            last_epoch = check_epoch
            last_checkpoint = check_point

    if len(last_checkpoint) != 0:
        return torch.load(last_checkpoint)
    else:
        return 0

def get_SegThor_regions():
    regions = {
        "esophagus": 1,
        "heart": 2,
        "trachea": 3,
        "aorta": 4
    }
    return regions


def get_LCTSC_regions():
    regions = {
        "Esophagus": 1,
        "Spinalcord": 2,
        "Heart": 3,
        "Left-lung": 4,
        "Right-lung": 5
    }
    return regions

def save_validation_nifti(img, gt, seg, path, patient, affine, model_name, epoch):
    seg = np.transpose(seg, (2, 1, 0))
    new_img = nib.Nifti1Image(seg, affine)
    nib.save(new_img, path + f"/{model_name}_e_{epoch}_P_{patient}_mask.nii.gz")
    if model_name == "Attention_Unet":
        img = np.transpose(img, (2, 1, 0))
        new_img = nib.Nifti1Image(img,affine)
        nib.save(new_img, path + f"/{model_name}_e_{epoch}_P_{patient}.nii.gz")
        gt = np.transpose(gt, (2, 1, 0))
        new_img = nib.Nifti1Image(gt,affine)
        nib.save(new_img,path+f"/{model_name}_e_{epoch}_P_{patient}_GT.nii.gz")