import torch
import cv2
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import Dataset, DataLoader
from custom_utils import get_train_transforms, get_val_transforms, collate_fn, show_batch
from config import RESIZE_HEIGHT, RESIZE_WIDTH, RIP_PATH, NO_RIP_PATH, LABELS_PATH, CLASSES, TAR_PATH, BATCH_SIZE, NUM_WORKERS


class RipDetectorDataset(Dataset):

    def __init__(self, img_txt, labels_csv, height, width, classes, transforms=None):
        my_file = open(img_txt, "r")
        data = my_file.read()
        self.img_list = data.replace('\n', ' ').split(" ")[:-1]
        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df.drop(labels_df.index[339, 1081])
        labels_df = labels_df.drop(labels_df.index[1081])
        self.labels_df = labels_df.drop_duplicates(subset=['Name'])
        self.img_height = height
        self.img_width = width
        self.cls = classes
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        image = cv2.imread(img_path)

        # convert resize and scale intensity
        image = image.astype(np.float32)
        image_resize = cv2.resize(image, (self.img_width, self.img_height))
        image_resize /= 255

        # extract image annotations
        img_df = self.labels_df.loc[self.labels_df['Name'] == os.path.basename(img_path), ['x1', 'y1', 'x2', 'y2', 'label']]
        bbox_label = img_df.to_numpy()
        if bbox_label.size > 0:
            bboxes, clss = bbox_label[:, 0:4], bbox_label[:, 4]
        else:
            bboxes = np.array((0, 0, self.img_width, self.img_height)).reshape(1, 4)
            clss = np.array(0).reshape(1, 1)

        # get the original image height and width
        image_width = image.shape[1]
        image_height = image.shape[0]

        if bbox_label.size > 0:
            # resize bbox
            bboxes[:, [0, 2]] = self.img_width * bboxes[:, [0, 2]] / image_width
            bboxes[:, [1, 3]] = self.img_height * bboxes[:, [1, 3]] / image_height
            clss = clss.reshape(len(clss), 1)

        area = torch.as_tensor((bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]))
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(clss, dtype=torch.int64)

        # prepare target dictionary
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels.squeeze(1)
        #target["area"] = area
        #target["image_id"] = os.path.basename(img_path)

        # apply image transform
        if self.transforms:
            sample = self.transforms(image=image_resize,
                                     bboxes=bboxes,
                                     labels=labels)
            image_resize = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        return image_resize, target


def create_train_dataset():
    return RipDetectorDataset('train.txt', LABELS_PATH, RESIZE_HEIGHT, RESIZE_WIDTH, CLASSES, get_train_transforms())

def create_valid_dataset():
    return RipDetectorDataset('val.txt', LABELS_PATH, RESIZE_HEIGHT, RESIZE_WIDTH, CLASSES, get_val_transforms())

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


if __name__=='__main__':

    train_set = create_train_dataset()
    print(f'Number of training images: {len(train_set)}')
    train_loader = create_train_loader(train_set)
    show_batch(train_loader, 3)