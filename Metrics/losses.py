import torch
from torch.nn.functional import one_hot
import torch.nn as nn


class hybrid_loss(nn.Module):
    def __init__(self, num_class, smooth=0.00001):
        super(hybrid_loss, self).__init__()
        self.smooth = smooth
        self.num_class = num_class

    def forward(self, true, pred):
        true = true.squeeze()
        true = one_hot(true, num_classes=self.num_class)

        pred = pred.argmax(dim=1).squeeze()
        pred = one_hot(pred, num_classes=self.num_class)

        assert true.shape == pred.shape

        dice_all = torch.zeros(1, device="cuda:0")
        iou_all = torch.zeros(1, device="cuda:0")
        print(dice_all.shape)

        for i in range(self.num_class):
            if true[:, :, i].max() > 0:
                y_pred = pred[:, :, i].contiguous().view(-1)
                y_true = true[:, :, i].contiguous().view(-1)

                intersection = torch.mul(y_pred, y_true).sum()

                dsc = torch.sub(1., torch.div(torch.add(torch.mul(torch.tensor([2.], device="cuda:0"), intersection), self.smooth), torch.add(y_pred.sum(), y_true.sum(), self.smooth)))
                iou = torch.sub(1., torch.div(torch.add(intersection, self.smooth), torch.sub(torch.add(pred.sum(), true.sum(), + self.smooth), intersection)))

                dice_all = torch.cat((dice_all, torch.tensor([dsc], device="cuda:0")))
                iou_all = torch.cat((iou_all, torch.tensor([iou], device="cuda:0")))
            else:
                continue

        return torch.add(torch.mean(dice_all[1:]),torch.mean(iou_all[1:]))

