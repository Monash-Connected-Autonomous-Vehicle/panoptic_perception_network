import torch
import torch.nn as nn



class Yolo_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # lambda constants
        self.lambda_class = 1
        self.lambda_noobj = 5
        self.lambda_box = 5
        self.lambda_obj = 1

        self.bce = nn.BCELoss()
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

        if self.CUDA:
            prediction = prediction.to(self.device)
            label = label.to(self.device)

        objs = (label[...,4] == 1)
        njs = (label[...,4] == 10)

        no_obj_loss = self.bce(
            (prediction[...,4][njs].float()), (label[..., 4][njs].float())
        )

        obj_loss = self.bce(
            (prediction[...,4][objs].float()), (label[...,4][objs].float())
        )

        bbox_loss = self.mse(
            (torch.sqrt(prediction[..., 0:4][objs])), (torch.sqrt(label[..., 0:4][objs]))
        )

        cls_argmaxs = torch.argmax(label[..., 5:], dim=-1)
        class_loss = self.ce(
            (prediction[...,5:][objs]), (cls_argmaxs[objs])
        )

        ## combine all losses
        loss = self.lambda_box*bbox_loss + self.lambda_obj*obj_loss + self.lambda_noobj*no_obj_loss + self.lambda_class*class_loss
        
        return loss