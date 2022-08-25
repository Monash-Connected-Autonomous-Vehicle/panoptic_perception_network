from pickletools import optimize
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
#from tqdm import tqdm
import wandb

from model import *
from dataset import DetectionDataset, Pad, ToTensor, Normalise
from loss import Yolo_Loss

wandb.init(project="yolov3-train-val-2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load data
label_dict = "/home/mcav/DATASETS/bdd100k/labels/bdd100k_labels_images_val.json"   # labels json 
root_dir = "/home/mcav/DATASETS/bdd100k/images/100k/val"                # images file

# hyperparams
# val size: 10,000
bs = 16
learning_rate = 1e-3
n_epoch = 10

# set rgb mean and std for normalise
rgb_mean = [92.11938007161459, 102.83839236762152, 104.90335580512152]
rgb_std  = [66.09941202519124, 70.6808655565459, 75.05305001603533]

## Load custom dataset + transforms
print("Loading Dataset...")
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
    shuffle=True
    #num_workers=1
)
print("Done.")

## Define network
print("Loading Network...")
net = Net(cfgfile="cfg/model.cfg").to(device)

## Define Loss Function and Optimiser
criterion = Yolo_Loss()
optimizer = optim.SGD(
    params=net.parameters(), 
    lr=learning_rate, 
    momentum=0.9
    )
print("Done.")

## Train network
CUDA = torch.cuda.is_available()
all_losses = []



wandb.config = {
    "learning_rate": learning_rate,
    "epochs": n_epoch,
    "batch_size": bs
}

torch.autograd.set_detect_anomaly(True)

print("Training...")
for epoch in range(n_epoch): # each image gets 3 detections, this happens n_epoch times

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        input_img, labels = data.values()
        optimizer.zero_grad()

        # torch.cuda.amp.autocast()
        # forward pass
        outputs = net(input_img.to(device), CUDA)
        # compute loss
        loss = criterion(outputs, labels).float()
        
        # back prop      
        loss.backward()
        optimizer.step()
        #print(loss)
        
        # print stats
        running_loss +=loss.item()
        if i % bs == bs-1: # print every bs mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / bs:.3f}')
            all_losses.append(running_loss / bs)
            wandb.log({"loss": running_loss / bs})
            running_loss = 0.0




print("Training complete.")

# save weights
torch.save(net.state_dict(), "weights/100_images.weights")