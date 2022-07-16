# Standard library imports
import os 
from collections import OrderedDict

# Third party imports
import numpy as np
import torch
import torch.nn.functional as f
import math
import cv2

# Local imports
from keypointEstimators.models.wholepose.config import cfg
from keypointEstimators.models.wholepose.pose_hrnet import get_pose_net


def pose_process(coords, hms):
    # hms: num_joints x h x w numpy array
    # coords: num_joints x 2 numpy array
    hm_w = hms.shape[2]
    hm_h = hms.shape[1]
    # score = np.zeros((hms.shape[0],1), dtype=np.float32)

    # print(hm_w, ', ', hm_h)
    for p in range(coords.shape[0]):
        hm = hms[p]
        
        px = int(math.floor(coords[p][0] + 0.5))
        py = int(math.floor(coords[p][1] + 0.5))
        # score[p] = hm[py, px]
        coords[p][2] = hm[py, px]
        if 1 < px < hm_w -1 and 1 < py < hm_h - 1:
            diff = np.array(
                [hm[py][px+1] - hm[py][px-1], hm[py+1][px] - hm[py-1][px]]
            )
            # print(px, ', ', py)
            # print(diff[0], ', ', diff[1])
            coords[p][:2] += np.sign(diff) * .25
    return coords

def norm_numpy_totensor(img):

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    img = img.astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2)

def stack_flip(img):
    img_flip = cv2.flip(img, 1)
    return np.stack([img, img_flip], axis=0)

def merge_hm(hms_list):
    index_mirror = np.concatenate([
                [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                [21,22,23,18,19,20],
                np.arange(40,23,-1), np.arange(50,40,-1),
                np.arange(51,55), np.arange(59,54,-1),
                [69,68,67,66,71,70], [63,62,61,60,65,64],
                np.arange(78,71,-1), np.arange(83,78,-1),
                [88,87,86,85,84,91,90,89],
                np.arange(113,134), np.arange(92,113)
                ]) - 1

    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1,:,:,:] = torch.flip(hms[1,index_mirror,:,:], [2])
    
    hm = torch.cat(hms_list, dim=0)
    # print(hm.size(0))
    hm = torch.mean(hms, dim=0)
    return hm

def model_init():

    config = 'keypointEstimators/models/wholepose/wholebody_w48_384x288.yaml'
    cfg.merge_from_file(config)

    # dump_input = torch.randn(1, 3, 256, 256)
    # newmodel = PoseHighResolutionNet()
    newmodel = get_pose_net(cfg, is_train=False)
    #print(newmodel)
    # dump_output = newmodel(dump_input)
    # print(dump_output.size())
    checkpoint = torch.load('keypointEstimators/models/wholepose/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth')

    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'backbone.' in k:
            name = k[9:] # remove module.
        if 'keypoint_head.' in k:
            name = k[14:] # remove module.
        new_state_dict[name] = v
    newmodel.load_state_dict(new_state_dict)

    newmodel.cpu().eval()

    return newmodel

def frame_process(wholepose, frame):

    multi_scales = [512,640]
    
    frame_height, frame_width = frame.shape[:2]
    
    maxSize = min(frame_width, frame_height)

    if maxSize < 256 or maxSize < 512 - 256/2:
        frame_height = 256
        frame_width = 256
    else:
        frame_height = 512
        frame_width = 512
    
    frame = cv2.resize(frame, (frame_height,frame_width))

    out = []

    for scale in multi_scales:
        if scale != 512:
            img_temp = cv2.resize(frame, (scale,scale))
        else:
            img_temp = frame
        img_temp = stack_flip(img_temp)
        img_temp = norm_numpy_totensor(img_temp).cpu()
        hms = wholepose(img_temp)
        if scale != 512:
            out.append(f.interpolate(hms, (frame_width // 4,frame_height // 4), mode='bilinear'))
        else:
            out.append(hms)

    out = merge_hm(out)

    result = out.reshape((133,-1))
    result = torch.argmax(result, dim=1)

    result = result.cpu().numpy().squeeze()

    y = result // (frame_width // 4)
    x = result % (frame_width // 4)
    pred = np.zeros((133, 3), dtype=np.float32)
    pred[:, 0] = x
    pred[:, 1] = y

    hm = out.cpu().detach().numpy().reshape((133, frame_height//4, frame_height//4))

    pred = pose_process(pred, hm)
    pred[:,:2] *= 4.0
    
    return pred[:,:2]/256
    
