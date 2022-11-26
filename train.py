from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS,
    RIP_PATH, NO_RIP_PATH, TAR_PATH, PERCENT_TRAIN
)
from model import create_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot, create_txt
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def train(train_loader, model, epoch):
    global train_itr
    global train_loss_list

    prog_bar = tqdm(train_loader, total=(len(train_loader)), desc=f"Train Epoch {epoch}", unit="Batch")
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_postfix(loss=loss_value)
    return train_loss_list


def validation(val_loader, model, epoch):
    global val_itr
    global val_loss_list

    prog_bar = tqdm(val_loader, total=(len(val_loader)), desc=f"Validation Epoch {epoch}", unit="Batch")
    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_postfix(loss=loss_value)
    return val_loss_list


if __name__ == '__main__':

    create_txt(RIP_PATH, NO_RIP_PATH, TAR_PATH, PERCENT_TRAIN,
               'C:\\Giora\\TAU\\MSc_courses\\Deep_Learning\\final_project\\training_data\\with_rips\\rip-10.png', 4)

    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()

    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    train_loss_epc_list = []
    val_loss_epc_list = []

    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_batch
        show_batch(next(iter(train_loader)), 2)

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    for epoch in range(NUM_EPOCHS):
        train_loss_hist.reset()
        val_loss_hist.reset()

        train_loss = train(train_loader, model, epoch)
        val_loss = validation(valid_loader, model, epoch)

        # save the best model till now if we have the least loss in the...
        # ... current epoch
        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )
        # save the current epoch model
        save_model(epoch, model, optimizer)
        # save loss plot
        save_loss_plot(OUT_DIR, train_loss, val_loss, 'iterations')

        train_loss_epc_list.append(train_loss_hist.value)
        val_loss_epc_list.append(val_loss_hist.value)
        save_loss_plot(OUT_DIR, train_loss_epc_list, val_loss_epc_list, 'epoch')
