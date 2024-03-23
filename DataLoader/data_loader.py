import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
from PIL import Image

# from DataLoader.augmentation import augment_image_label


class OrganSegmentationDataset(Dataset):
    def __init__(
            self,
            images_dir,
            test_subset = None,
            subset="train",
            train_st=-1,
            test_st=-1
    ):
        self.subset = subset
        self.required_test = False

        if subset == 'train':
            self.images_dir = os.path.join(images_dir, "TrainDataset", "image")
            self.masks_dir = os.path.join(images_dir, "TrainDataset", "masks")
        else:
            self.images_dir = os.path.join(images_dir, "TestDataset", test_subset, "images")
            self.masks_dir = os.path.join(images_dir, "TestDataset", test_subset, "masks")

        print(os.path.join(self.images_dir,"*.png"))
        print(os.path.join(self.masks_dir,"*.png"))
        self.image_paths = glob.glob(self.images_dir + os.sep + "*.png")
        self.masks_paths = glob.glob(self.masks_dir + os.sep + "*.png")

        assert len(self.image_paths) == len(self.masks_paths)

        print("reading {} images...".format(subset))
        print("Count ", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, id):
        image_size = 256
        file_path = self.image_paths[id]
        mask_path = self.masks_paths[id]
        file_name = file_path.split(os.sep)[-1].split(".")[0]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 70, 1, cv2.THRESH_BINARY)
        mask = np.array(Image.fromarray(mask, 'L').resize((image_size, image_size), Image.Resampling.BILINEAR))

        image = cv2.imread(file_path)
        image = np.array(Image.fromarray(image, 'RGB').resize((image_size, image_size), Image.Resampling.BILINEAR))
        image = image.transpose(2, 0, 1)

        # image, mask = augment_image_label(np.squeeze(image), np.squeeze(mask), 256,
        #                                   trans_threshold=.6, horizontal_flip=None, rotation_range=30,
        #                                   height_shift_range=0.05, width_shift_range=0.05,
        #                                   shear_range=None, zoom_range=(0.95, 1.05), elastic=None, add_noise=0.0)

        # normalization of image
        # min_value = np.amin(image)
        # image = image - min_value
        max_value = np.amax(image)
        image = image / max_value

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(int))
        mask_tensor = mask_tensor.long()
        # print(f'Patient: {patient_id} Splice: {slice_id}')

        return image_tensor, mask_tensor, file_name


def data_loaders(data_dir, test_subset):
    dataset_train, dataset_valid = datasets(data_dir, test_subset)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(data_dir, test_subset):
    train = OrganSegmentationDataset(images_dir= data_dir,
                                     subset="train"
                                     )
    valid = OrganSegmentationDataset(images_dir=data_dir,
                                     test_subset = test_subset,
                                     subset="test"
                                     )
    return train, valid
