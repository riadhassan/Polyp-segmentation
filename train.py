import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from DataLoader.data_loader import data_loaders
import argparse
from Network import Network_wrapper
from Metrics import losses
from Metrics import Evaluation_metric as evaluate
import random
import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss
from torch.nn.functional import one_hot


def conf():
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str,
                      default="C:\\Users\\riad_\\OneDrive\\Desktop\\Demo_dataset\\polyp dataset")
    args.add_argument("--output_path", type=str,
                      default="C:\\Users\\riad_\\OneDrive\\Desktop\\Polyp")
    args.add_argument("--dataset", type=str, default="Combined")
    args.add_argument("--test_subset", type=str, default="CVC-ClinicDB")
    args.add_argument("--model_name", type=str, default="our")
    args.add_argument("--epoch_num", type=int, default=200)
    args = args.parse_args()
    return args

def out_directory_create(conf):
    if not os.path.exists(os.path.join(conf.output_path, conf.dataset)):
        os.mkdir(os.path.join(conf.output_path, conf.dataset))

    if not os.path.exists(os.path.join(conf.output_path, conf.dataset, conf.model_name)):
        os.mkdir(os.path.join(conf.output_path, conf.dataset, conf.model_name))

    if not os.path.exists(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Model")):
        os.mkdir(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Model"))

    if not os.path.exists(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Image")):
        os.mkdir(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Image"))

    return os.path.join(conf.output_path, conf.dataset, conf.model_name)

def main(conf):
    dice_CE = DiceCELoss(softmax=True)
    IoU = DiceLoss(jaccard=True, softmax=True)
    loss_function_CE = nn.CrossEntropyLoss()
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    loader_train, loader_valid = data_loaders(data_dir = conf.data_path, test_subset = conf.test_subset)
    loaders = {"train": loader_train, "valid": loader_valid}

    out_channel = 2

    model = Network_wrapper.model_wrapper(conf)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loss = losses.hybrid_loss(out_channel)

    output_directory = out_directory_create(conf)

    loss_train = []
    for epoch in tqdm(range(conf.epoch_num)):
        print("\n {epc} is running".format(epc=epoch))
        result = {"dice":[], "hd":[], "iou":[], "asd":[], "assd":[]}
        img_print = random.randint(500, 600)

        for phase in ['valid']:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for i, data in enumerate(loaders[phase]):
                x, y_true, file_name = data
                x, y_true = x.to(device), y_true.to(device)
                print(x.shape)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred1, y_pred2, y_pred3, refined  = model(x)
                    y_true_one_hot = one_hot(y_true, num_classes=out_channel).permute(0, 3, 1, 2)

                    loss1 = dice_CE(y_pred1, y_true_one_hot)
                    loss2 = dice_CE(y_pred2, y_true_one_hot)
                    loss3 = dice_CE(y_pred3, y_true_one_hot)
                    loss4 = dice_CE(refined, y_true_one_hot)

                    loss = loss1 + loss2 + loss3 + loss4

                    if i%100 == 0:
                        print(f"Iteration: {i} Loss: {loss}")

                    if phase == "valid":
                        y_pred = refined.argmax(dim=1)
                        dice, hd, iou, asd, assd = evaluate.evaluate_case(y_pred, y_true, out_channel)

                        result["dice"].append(dice.item())
                        result["hd"].append(hd.item())
                        result["iou"].append(iou.item())
                        result["asd"].append(asd.item())
                        result["assd"].append(assd.item())

                        print(result)

                        if i % img_print == 0:
                            y_pred_np = y_pred.detach().cpu().numpy().squeeze()
                            y_true_np = y_true.detach().cpu().numpy().squeeze()
                            main_image_np = x.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
                            plt.figure(i)
                            plt.subplot(1, 3, 1)
                            plt.axis('off')
                            plt.imshow(main_image_np)
                            plt.subplot(1, 3, 2)
                            plt.axis('off')
                            plt.imshow(y_true_np)
                            plt.subplot(1, 3, 3)
                            plt.axis('off')
                            plt.imshow(y_pred_np)
                            plt.savefig(os.path.join(output_directory, "Image", f"{file_name[0]}"), bbox_inches='tight')
                            plt.close()

                    elif phase == "train":
                        loss_train.append(loss)
                        loss.backward()
                        optimizer.step()
        average_result = {}
        for metric in result.keys():
            data = result[metric]
            average_result[metric] = sum(data)/len(data)

        print(f"Epoch: {epoch} Evaluation of {conf.model_name} Validate ")

        if epoch < 1:
            prev_mean_dice = 0

        curr_mean_dice = torch.mean(average_result["dice"])
        model_name = f"{conf.model_name}_epoch_{epoch}.pth"
        checkpoint_file = os.path.join(output_directory, "Model", model_name)
        if curr_mean_dice > prev_mean_dice:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_file)
            prev_mean_dice = curr_mean_dice
        else:
            continue

if __name__ == "__main__":
    main(conf())
