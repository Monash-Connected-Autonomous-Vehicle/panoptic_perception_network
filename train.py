import torch
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
#from tqdm import tqdm
import time
import wandb
import argparse

from model import *
from dataset import DetectionDataset, Pad, ToTensor, Normalise
from loss import Yolo_Loss

def arg_parse():
    parser = argparse.ArgumentParser(description="YOLOv3 Train Module")

    parser.add_argument(
        "--bs",
        dest="bs",
        help="Batch size.",
        default=16
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        help="Learning rate.",
        default=1e-4
    )
    parser.add_argument(
        "--wd",
        dest="wd",
        help="Weight decay for ADAM.",
        default=1e-3
    )
    parser.add_argument(
        "--ep",
        dest="ep",
        help="Number of training epochs.",
        default=10
    )
    parser.add_argument(
        "--opt",
        dest="opt",
        help="Training optimiser - SGD or ADAM",
        default="ADAM",
        type=str
    )
    parser.add_argument(
        "--wb",
        dest="wb",
        help="Logging with weights and biases [1: True, 0: False]",
        default=1,
    )

    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load data
label_dict = "/home/mcav/DATASETS/bdd100k/labels/bdd100k_labels_images_val.json"   # labels json 
root_dir = "/home/mcav/DATASETS/bdd100k/images/100k/val"                # images file

# hyperparams
# val size: 10,000
args = arg_parse()
bs = int(args.bs)
learning_rate = float(args.lr)
weight_decay = float(args.wd)
n_epoch = int(args.ep)
opt = args.opt
wb = bool(int(args.wb))

print("CHOSEN ARGS")
print("----------------------------------------------------------")
print(f"Batch size:\t{bs}")
print(f"Learning rate:\t{learning_rate}")
print(f"Weight decay:\t{weight_decay}")
print(f"Epochs:\t\t{n_epoch}")
print(f"Optimiser:\t{opt}")
print(f"Log metrics:\t{wb}")
print("----------------------------------------------------------")

if wb:
    wandb.init(project="yolov3-train-val-5")
    wandb.config.batch_size = bs
    wandb.config.lr = learning_rate
    wandb.config.weight_decay = weight_decay
    wandb.config.n_epoch = n_epoch
    wandb.config.optimizer = opt


# set rgb mean and std for normalise
rgb_mean = [92, 103, 105]
rgb_std  = [66, 71, 75]
# rgb_mean = [92.11938007161459, 102.83839236762152, 104.90335580512152]
# rgb_std  = [66.09941202519124, 70.6808655565459, 75.05305001603533]

## Load custom dataset + transforms
print("----------------------------------------------------------")
print("Loading Dataset...")
load_dataset_start = time.time()

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
    num_workers=8,
    pin_memory=True,
)
load_dataset_end = time.time()
print(f"Done in {load_dataset_end - load_dataset_start:.3f}s.")
print("----------------------------------------------------------")

## Define network
print("Loading Network, Loss and Optimiser...")
load_net_start = time.time()

net = Net(cfgfile="cfg/model.cfg").to(device)

## Define Loss Function and Optimiser
criterion = Yolo_Loss()
if opt == "SGD":
    optimizer = optim.SGD(
        params=net.parameters(), 
        lr=learning_rate, 
        momentum=0.9
        )
elif opt == "ADAM":
    optimizer = optim.Adam(
        params=net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
else:
    print("Only [SGD] and [ADAM] optimisers have been configured.")

load_net_end = time.time()
print(f"Done in {load_net_end - load_net_start:.3f}s.")
print("----------------------------------------------------------")

CUDA = torch.cuda.is_available()
all_losses = []


## TURN ON FOR DEBUGGING
#torch.autograd.set_detect_anomaly(True)

scaler = torch.cuda.amp.GradScaler()

## Train network
print("Training...")
for epoch in range(n_epoch): # each image gets 3 detections, this happens n_epoch times

    running_loss = 0.0
    running_no_obj_loss = 0.0
    running_obj_loss = 0.0
    running_bbox_loss = 0.0
    running_class_loss = 0.0
    running_class_acc = 0.0
    running_objs_acc = 0.0
    running_njs_acc = 0.0
    running_iou = 0.0

    epoch_start = time.time()

    for i, data in enumerate(train_loader):
        input_img, labels = data.values()
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():
        # forward pass
        outputs = net(input_img.to(device), CUDA)
        # compute loss
        loss, no_obj_loss, obj_loss, bbox_loss, class_loss, cls_acc, objs_acc, njs_acc, iou = criterion(outputs, labels)
        
        # # without scaler     
        # loss.float().backward()
        # optimizer.step()

        # with scaler        
        scaler.scale(loss.float()).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # log and print stats
        running_loss += loss.item()
        running_no_obj_loss += no_obj_loss.item()
        running_obj_loss += obj_loss.item()
        running_bbox_loss += bbox_loss.item()
        running_class_loss += class_loss.item()
        running_class_acc += cls_acc.item()
        running_objs_acc += objs_acc.item()
        running_njs_acc += njs_acc.item()
        running_iou += iou.item()

        if i % bs == bs-1: # print every bs mini-batches
            epoch_end = time.time()
            print(f'[{epoch + 1}, {i + 1:5d}]\tloss: {running_loss / bs:.3f}\ttime: {(epoch_end - epoch_start) / bs:.3f}s')
            all_losses.append(running_loss / bs)
            if wb:
                wandb.log({
                    "loss": running_loss / bs,
                    "no_obj_loss": running_no_obj_loss / bs,
                    "obj_loss": running_obj_loss / bs,
                    "bbox_loss": running_bbox_loss / bs,
                    "class_loss": running_class_loss / bs,
                    "class_acc": running_class_acc / bs,
                    "objs_acc": running_objs_acc / bs,
                    "njs_acc": running_njs_acc / bs,
                    "iou": running_iou / bs
                })

            # reset
            running_loss = 0.0
            running_no_obj_loss = 0.0
            running_obj_loss = 0.0
            running_bbox_loss = 0.0
            running_class_loss = 0.0
            running_class_acc = 0.0
            running_objs_acc = 0.0
            running_njs_acc = 0.0
            running_iou = 0.0
    # save weights at the end of each epoch
    torch.save(net.state_dict(), f"weights/during_run/epoch_{epoch}.weights")



print("Training complete.")

# save weights
torch.save(net.state_dict(), f"weights/{opt}_{bs}_{learning_rate}_{n_epoch}_val.weights")