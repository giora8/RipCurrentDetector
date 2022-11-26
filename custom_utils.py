import albumentations as A
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import shuffle
from albumentations.pytorch import ToTensorV2
from config import CLASSES
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn")


plt.style.use('ggplot')


class Averager():
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total =+ value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel():
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/best_model.pth')


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def get_train_transforms():
    """
    :return: transformations to apply on training set
    """
    return A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
        ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']}
    )


def get_val_transforms():
    """
    :return: transformations to apply on validation/ test sets
    """
    return A.Compose([
        ToTensorV2(p=1.0),
        ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']}
    )


def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/last_model.pth')


def save_loss_plot(OUT_DIR, train_loss, val_loss, xlabel):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel(xlabel)
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel(xlabel)
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss_" + xlabel +".png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss_" + xlabel +".png")
    plt.close('all')


def show_batch(batch, N_show):

    if len(batch) > 0:
        N = np.min((N_show, len(batch[0])))
        images = batch[0]
        targets = batch[1]
        for i in range(N):
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box_num, box in enumerate(boxes):
                if labels[box_num] > 0:
                    cv2.rectangle(sample,
                                  (box[0], box[1]),
                                  (box[2], box[3]),
                                  (0, 0, 255), 2)

                    cv2.putText(sample, CLASSES[labels[box_num]],
                                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def create_txt(with_rips_path, no_rip_path, target_path, train_percent, specific_img=None, N=None):

    if not specific_img:
        img_with_rip = os.listdir(with_rips_path)
        img_no_rip = os.listdir(no_rip_path)
        shuffle(img_with_rip)
        shuffle(img_no_rip)
        train_rips = img_with_rip[0:int(train_percent*len(img_with_rip))]
        train_no_rips = img_no_rip[0:int(train_percent * len(img_no_rip))]

        train_rips = [os.path.join(with_rips_path, z) for z in train_rips]
        train_no_rips = [os.path.join(no_rip_path, z) for z in train_no_rips]

        val_rips = img_with_rip[int(train_percent*len(img_with_rip)):]
        val_no_rips = img_no_rip[int(train_percent * len(img_no_rip)):]

        val_rips = [os.path.join(with_rips_path, z) for z in val_rips]
        val_no_rips = [os.path.join(no_rip_path, z) for z in val_no_rips]

        train_list = train_rips + train_no_rips
        val_list = val_rips + val_no_rips

        shuffle(train_list)
        shuffle(val_list)

        with open(os.path.join(target_path, 'train.txt'), 'w') as f:
            for img in train_list:
                f.write(img+"\n")
        with open(os.path.join(target_path, 'val.txt'), 'w') as f:
            for img in train_list:
                f.write(img+"\n")
    else:
        l = [specific_img] * N
        with open(os.path.join(target_path, 'train.txt'), 'w') as f:
            for img in l:
                f.write(img+"\n")
        with open(os.path.join(target_path, 'val.txt'), 'w') as f:
            for img in l:
                f.write(img+"\n")
