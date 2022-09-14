import torch
import torch.nn as nn

class Yolo_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # lambda constants
        self.LAMBDA_CLS = 1
        self.LAMBDA_NO_OBJ = 5
        self.LAMBDA_BBOX = 1
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
        
        # print(f"pred:   {prediction}")
        # print(f"label:  {label}")

        # if GPU exists, move tensors there
        if self.CUDA:
            prediction = prediction.to(self.device)
            label = label.to(self.device)

        # create masks for object and no object
        objs = (label[...,4] == 1)
        njs = (label[...,4] == 0)

        # separate out components of prediction tensor
        pred_obj_prob = prediction[...,4].float()
        pred_bbox_centre = prediction[..., 0:2]
        pred_bbox_dims = prediction[..., 2:4]
        pred_cls_logits = prediction[..., 5:]

        #print(pred_obj_prob)

        # separate out components of label tensor
        label_obj_prob = label[...,4].float()
        label_bbox_centre = label[..., 0:2]
        label_bbox_dims = label[..., 2:4]
        label_cls_logits = label[..., 5:]

        ################################################################################

        # ACCURACY FOR NO OBJ and OBJ 
        pred_njs = pred_obj_prob[njs] # these ones should == 0
        pred_objs = pred_obj_prob[objs] # these ones should == 1

        label_njs = label_obj_prob[njs]
        label_objs = label_obj_prob[objs]

        # hardmax predictions
        pred_njs_hm = (pred_njs > 0.5).float()
        pred_objs_hm = (pred_objs > 0.5).float()

        # calculate accuracies
        objs_acc = torch.sum(pred_objs_hm == label_objs)/label_objs.shape[0]*100
        njs_acc = torch.sum(pred_njs_hm == label_njs)/label_njs.shape[0]*100

        ################################################################################

        # NO OBJECT LOSS

        # print("pred_obj_prob max, min")
        # print(pred_obj_prob[njs].max().item(), pred_obj_prob[njs].min().item())
        # print("label_obj_prob max, min")
        # print(label_obj_prob[njs].max().item(), label_obj_prob[njs].min().item())
        
        no_obj_loss = self.bce(
            (pred_obj_prob[njs]), (label_obj_prob[njs])
        )
        #print(no_obj_loss)

        # OBJECT LOSS
        obj_loss = self.bce(
            (pred_obj_prob[objs]), (label_obj_prob[objs])
        )
        #print(obj_loss)

        ################################################################################

        # BBOX LOSS
        bbox_centre_loss = self.mse(
            (torch.sqrt(pred_bbox_centre[objs])), (torch.sqrt(label_bbox_centre[objs]))
        )
        bbox_dims_loss = self.mse(
            (torch.sqrt(pred_bbox_dims[objs])), (torch.sqrt(label_bbox_dims[objs]))
        )
        bbox_loss = bbox_centre_loss + bbox_dims_loss
        #print(bbox_loss)

        ################################################################################

        # BBOX IOU

        pred_box = prediction[...,0:4]
        label_box = label[...,0:4]

        crn_pred_box = self.centre_dims_to_corners(pred_box[objs])
        crn_label_box = self.centre_dims_to_corners(label_box[objs])

        iou = self.bbox_iou(crn_pred_box, crn_label_box)

        batch_iou = torch.max(iou)*100

        ################################################################################

        # CLASS LOSS
        # need to get idx of correct class for each tensor
        # this is just how nn.CrossEntropyLoss() inputs labels 
        cls_argmaxs = torch.argmax(label_cls_logits, dim=-1)
        class_loss = self.ce(
            (pred_cls_logits[objs]), (cls_argmaxs[objs])
        )
        #print(class_loss)

        ################################################################################

        # CLASS ACCURACY FOR DEBUGGING
        cls_pred_argmaxs = torch.argmax(pred_cls_logits, dim=-1)
        cls_acc = (torch.sum(cls_argmaxs[objs] == cls_pred_argmaxs[objs])/cls_argmaxs[objs].shape[0])*100
        
        ################################################################################

        ## combine all losses
        loss = self.LAMBDA_BBOX*bbox_loss + self.LAMBDA_OBJ*obj_loss + self.LAMBDA_NO_OBJ*no_obj_loss + self.LAMBDA_CLS*class_loss
        
        return loss, no_obj_loss, obj_loss, bbox_loss, class_loss, cls_acc, objs_acc, njs_acc, batch_iou
    
    def centre_dims_to_corners(self, bbox):
        """Converts bbox attributes of form [x_centre, y_centre, width, height] to form [x1, y1, x2, y2]. 
        
        Use on an array of bboxes: [[bbox_1], [bbox_2], ... [bbox_n]].

        This form is used for easily calculating 2 bbox's IoU.

        Args:
            bbox (np.ndarray): Bbox centre and dims [x_centre, y_centre, width, height].

        Returns:
            new_bbox (np.ndarray): Bbox corner coords [x1, y1, x2, y2].
        """
        eps = 1e-4 # add eps in the case width, height is tiny, resulting in no box and nan iou

        x_c, y_c, w, h = bbox[...,0], bbox[...,1], bbox[...,2]+eps, bbox[...,3]+eps

        x1, x2 = x_c-(w/2), x_c+(w/2)
        y1, y2 = y_c-(h/2), y_c+(h/2)
        

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        y1 = torch.unsqueeze(y1, 1)
        y2 = torch.unsqueeze(y2, 1)

        new_bbox = torch.cat((x1, y1, x2, y2), axis=1)

        return new_bbox

    def bbox_iou(self, box1, box2):
        """Returns intersection over union of two bounding boxes.

        Strictly performed on tensors.

        Args:
            box1 (torch.FloatTensor): Coordinates of bbox 1.
            box2 (torch.FloatTensor): Coordinates of bbox 2.

        Returns:
            iou (float): IOU of two input bboxes.
        """
        # get coords of bboxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

        # get coords of intersection
        intersect_x1 = torch.max(b1_x1, b2_x1)
        intersect_y1 = torch.max(b1_y1, b2_y1)
        intersect_x2 = torch.min(b1_x2, b2_x2)
        intersect_y2 = torch.min(b1_y2, b2_y2)

        # intersection area
        # clamp to > 0
        # this avoids areas being calculated for boxes with zero intersect
        intersect_area = torch.clamp(intersect_x2 - intersect_x1, min=0)*torch.clamp(intersect_y2 - intersect_y1, min=0)

        # union area
        b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)
        union_area = b1_area + b2_area - intersect_area

        # compute iou
        iou = intersect_area/union_area

        return iou    