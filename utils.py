from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, in_dims, anchors, n_classes, CUDA=True):
    """
    Converts detection feature map into 2D tensor, where each row are attributes of bbox.
    """
    batch_size = prediction.size(0)
    stride = in_dims // prediction.size(2)
    grid_size = prediction.shape[2]
    bbox_attrs = 5 + n_classes
    n_anchors = len(anchors)

    # reshaping
    prediction = prediction.view(batch_size, bbox_attrs*n_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*n_anchors, bbox_attrs)

    # divide anchors by stride of detection feature map as input image is 
    # larger than detection map by a factor of stride
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    ## transform output
    # apply sigmoid to x,y coords and object confidence score
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # add grid offset to centre coord predictions
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    # convert to float and reshape into column vectors
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, n_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    ## apply anchors to dimensions of bbox
    # log transform height and width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # apply sigmoid activation to class scores
    prediction[:,:,5:5+n_classes] = torch.sigmoid((prediction[:,:,5:5+n_classes]))

    # resize detection map to size of input image
    # bbox attributes are sized according to feature map e.g. 13x13
    # if input image is 416x416, multiply attributes by 32, or stride
    prediction[:,:,:4] *= stride

    return prediction

def write_results(prediction, confidence, n_classes, nms_conf=0.4):
    """
    Objectness score thresholding and non-max suppression.

    Input:
    =prediction= tensor of size (batch_size) x (n_bboxes) x (4 bbox attrs + 1 obj score + 80 class scores)
    """

    ## object confidence thresholding

    # prediction tensor contains info about Bx10647 bboxes
    # for each bbox that has an objectness score below threshol:
    # we set the value of its attr *entire row representing bbox) to zero
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    ## performing non-max suppression

    # bbox attrs described by centre coords
    # convert (centre_x, centre_y, height, width) -> (min_x, min_y, max_x, max_y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    # replace in the prediction tensor
    prediction[:,:,:4] = box_corner[:,:,:4]
    