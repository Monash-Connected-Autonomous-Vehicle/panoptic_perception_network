import torch
import torch.nn as nn

from utils import *

class Yolo_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # lambda constants
        self.LAMBDA_CLS = 1
        self.LAMBDA_NO_OBJ = 5
        self.LAMBDA_BBOX = 5
        self.LAMBDA_OBJ = 1

        self.bce = nn.BCELoss()
        self.bcewll = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

        self.CUDA = torch.cuda.is_available() # check CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get CUDA device

    def forward(self, prediction: torch.FloatTensor, label: torch.FloatTensor) -> float:
        """Computes loss between prediction and label.

        Args:
            prediction (torch.FloatTensor): Tensor of all prediction arrays of size 
                                                (n_batches, 10647, 5+n_classes).
            label (torch.FloatTensor): Tensor of all label arrays of size 
                                            (n_batches, 10647, 5+n_classes).

        Returns:
            loss (float): Total loss computed for this batch.
        """
        
        # if GPU exists, move tensors there
        if self.CUDA:
            prediction = predict_transform(prediction.to(self.device))
            label = label.to(self.device)

        # create masks for object and no object
        objs = (label[...,4] == 1)
        njs = (label[...,4] == 0)

        # separate out components of prediction tensor
        pred_obj_prob = prediction[...,4].float()
        pred_bbox_centre = prediction[..., 0:2]
        pred_bbox_dims = prediction[..., 2:4]
        pred_cls_logits = prediction[..., 5:]

        # separate out components of label tensor
        label_obj_prob = label[...,4].float()
        label_bbox_centre = label[..., 0:2]
        label_bbox_dims = label[..., 2:4]
        label_cls_logits = label[..., 5:]

        torch.save(label, "ex_tensors/loss_label.pt")
        torch.save(prediction, "ex_tensors/loss_pred.pt")
        # NO OBJECT LOSS
        no_obj_loss = self.bcewll(
            (pred_obj_prob[njs]), (label_obj_prob[njs])
        )
        #print(no_obj_loss)

        # OBJECT LOSS
        obj_loss = self.bcewll(
            (pred_obj_prob[objs]), (label_obj_prob[objs])
        )
        #print(obj_loss)

        # BBOX LOSS
        bbox_centre_loss = self.mse(
            (torch.sqrt(pred_bbox_centre[objs])), (torch.sqrt(label_bbox_centre[objs]))
        )
        bbox_dims_loss = self.mse(
            (torch.sqrt(pred_bbox_dims[objs])), (torch.sqrt(label_bbox_dims[objs]))
        )
        bbox_loss = bbox_centre_loss + bbox_dims_loss
        #print(bbox_loss)

        # CLASS LOSS
        # eed to get idx of correct class for each tensor
        # this is just how nn.CrossEntropyLoss() inputs labels 
        cls_argmaxs = torch.argmax(label_cls_logits, dim=-1)  
        class_loss = self.ce(
            (pred_cls_logits[objs]), (cls_argmaxs[objs])
        )
        #print(class_loss)

        ## combine all losses
        loss = self.LAMBDA_BBOX*bbox_loss + self.LAMBDA_OBJ*obj_loss + self.LAMBDA_NO_OBJ*no_obj_loss + self.LAMBDA_CLS*class_loss
        
        return loss