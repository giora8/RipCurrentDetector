import torch


BATCH_SIZE = 2
RESIZE_HEIGHT = 300  # resize the image height for training and transforms
RESIZE_WIDTH = 300  # resize the image width for training and transforms
NUM_EPOCHS = 10  # number of epochs to train for
NUM_WORKERS = 0
PERCENT_TRAIN = 0.95
#DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = 'cpu'

# rip\ no rip imgs path
RIP_PATH = 'C:\\Giora\\TAU\\MSc_courses\\Deep_Learning\\final_project\\training_data\\with_rips'
NO_RIP_PATH = 'C:\\Giora\\TAU\\MSc_courses\\Deep_Learning\\final_project\\training_data\\without_rip'
TAR_PATH = 'C:\\Giora\\Work\\MyOwnShit\\RipCurrentDetector'
# labels csv
LABELS_PATH = 'C:\\Giora\\TAU\\MSc_courses\\Deep_Learning\\final_project\\training_data\\data_labels.csv'
# classes: 0 index is reserved for background
CLASSES = [
    '__background__', '1'
]
NUM_CLASSES = len(CLASSES)

VISUALIZE_TRANSFORMED_IMAGES = True
# location to save model and plots
OUT_DIR = 'outputs'
