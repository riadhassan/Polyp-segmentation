import torch
from torch.nn.functional import one_hot
from monai.metrics import compute_average_surface_distance, compute_dice, compute_iou, compute_hausdorff_distance

def evaluate_case(image_gt, image_pred, class_num):
    assert image_gt.shape == image_pred.shape
    image_gt = image_gt[None, :, :, :]
    image_pred = image_pred[None, :, :, :]
    image_gt = one_hot(image_gt, num_classes=class_num).permute(0, 4, 1, 2, 3)
    image_pred = one_hot(image_pred, num_classes=class_num).permute(0, 4, 1, 2, 3)


    dc = torch.nan if torch.max(image_gt) < 1 or torch.max(image_pred) < 1 else compute_dice(image_pred, image_gt, include_background=False)
    hd = torch.nan if torch.max(image_gt) < 1 or torch.max(image_pred) < 1 else compute_hausdorff_distance(
        image_pred, image_gt)
    iou = torch.nan if torch.max(image_gt) < 1 or torch.max(image_pred) < 1 else compute_iou(image_pred, image_gt, include_background=False)
    asd = torch.nan if torch.max(image_gt) < 1 or torch.max(image_pred) < 1 else compute_average_surface_distance(
        image_pred, image_gt)
    assd = torch.nan if torch.max(image_gt) < 1 or torch.sum(image_pred) < 1 else compute_average_surface_distance(
        image_pred, image_gt, symmetric=True)

    return dc, hd, iou, asd, assd
