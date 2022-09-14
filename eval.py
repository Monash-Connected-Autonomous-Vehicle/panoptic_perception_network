import torch
import numpy as np
from model import Net
from dataset import DetectionDataset, Pad, ToTensor, Normalise
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bs = 16

label_dict = "/home/mcav/DATASETS/bdd100k/labels/bdd100k_labels_images_val.json"   # labels json 
root_dir = "/home/mcav/DATASETS/bdd100k/images/100k/val"                # images file

rgb_mean = [92.11938007161459, 102.83839236762152, 104.90335580512152]
rgb_std  = [66.09941202519124, 70.6808655565459, 75.05305001603533]

print("Loading Data")
transformed_train_data = DetectionDataset(
    label_dict=label_dict,                      # labels corresponding to images
    root_dir=root_dir,                          # images root dir
    classes_file="data/bdd100k.names",          # class names
    grid_sizes=[13, 26, 52],                    # grid sizes for detection
    anchors = np.array([                        # anchor box sizes per grid size
            [[116,90], [156,198], [373,326]],   
            [[30, 61], [62, 45], [59,119]],
            [[10, 13], [16, 30], [33, 23]],
        ]),
    transform=transforms.Compose([              # transforms
        Normalise(                              # 1. normalise
            mean=rgb_mean,                      
            std=rgb_std
        ),
        Pad(416),                               # 2. padding
        ToTensor()                              # 3. convert to tensor
    ])
)

# separate transformed dataset into batches
train_loader = DataLoader(
    transformed_train_data,
    batch_size=bs,
    shuffle=True,
    num_workers=8
)

print("Loading Network")
net = Net(cfgfile="cfg/model.cfg").to(device)

weights = "weights/ADAM_100_images.weights"
net.load_state_dict(torch.load(weights))
#net.eval()

CUDA = torch.cuda.is_available()

dataiter = iter(train_loader)
images, labels = dataiter.next()

