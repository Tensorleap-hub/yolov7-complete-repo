from typing import List, Union

import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse 
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from utils.datasets import create_dataloader
from leapcfg.config import CONFIG, hyp, data_dict
import torch
from models.yolo import Model
from utils.loss import ComputeLossOTA
import tensorflow as tf
import yaml

from leap_utils import bb_array_to_object
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from utils.general import non_max_suppression


nc = 1 if CONFIG['SINGLE_CLS'] else int(data_dict['nc'])  # number of classes
torch_model = Model(CONFIG['CFG'], ch=3, nc=nc, anchors=hyp.get('anchors'))
nl = torch_model.model[-1].nl
hyp['box'] *= 3. / nl  # scale to layers
hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
hyp['obj'] *= (CONFIG['IMGSZ'] / 640) ** 2 * 3. / nl  # scale to image size and layers
hyp['label_smoothing'] = CONFIG['LABEL_SMOOTHING']
torch_model.nc = nc  # attach number of classes to model
torch_model.hyp = hyp  # attach hyperparameters to model
torch_model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
compute_loss_ota = ComputeLossOTA(torch_model)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
# Preprocess Function


def preprocess_func() -> List[PreprocessResponse]:
    # Hyperparameters

    train_path = data_dict['train']
    test_path = data_dict['val']
    train_dataloader, dataset = create_dataloader(train_path, CONFIG['IMGSZ'], 1, CONFIG['GS'],
                                            Namespace(single_cls=CONFIG['SINGLE_CLS']),
                                            hyp=hyp, augment=False, cache=CONFIG['CACHE_IMAGES'], rect=CONFIG['RECT'],
                                            rank=-1, world_size=1, workers=CONFIG['WORKERS'],
                                            image_weights=False, quad=CONFIG['QUAD'], prefix='train: ')
    test_dataloader, dataset = create_dataloader(test_path, CONFIG['IMGSZ_TEST'], 1, CONFIG['GS'],\
                                                 Namespace(single_cls=CONFIG['SINGLE_CLS']),  # testloader
                                                hyp=hyp, cache=CONFIG['CACHE_IMAGES'], rect=True, rank=-1,
                                                world_size=1, workers=CONFIG['WORKERS'],
                                                pad=0.5, prefix='val: ')

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(length=len(train_dataloader), data={'dataset': train_dataloader.dataset})
    val = PreprocessResponse(length=len(test_dataloader), data={'dataset': test_dataloader.dataset})
    response = [train, val]
    return response


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image. 
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['dataset'][idx][0].permute((1, 2, 0)).numpy().astype('float32')/255.


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    #TODO - problem - can't have dynamic GT shape?
    #Remove none-zero gt before picking up
    torch_gt = preprocess.data['dataset'][idx][1]
    np_gt = np.zeros((CONFIG['MAX_INSTANCE'], 6))
    np_gt[:, 1] = nc + 1
    instances_count = torch_gt.shape[0]
    np_gt[:instances_count, :] = torch_gt[:min(CONFIG['MAX_INSTANCE'], instances_count), :]
    return np_gt


def arc_sigmoid(pred):
    return torch.log(pred / (1 - pred))


def transform_xy(pred, strides):
    x = [(pred[i][..., :2] / strides[i]
          + 0.5
          - torch_model.model[-1]._make_grid(*pred[i].shape[2:4]))/2 for i in range(len(pred))]
    for i in range(len(pred)):
        pred[i][..., :2] = arc_sigmoid(x[i])
    return pred


def transform_wh(pred):
    for i in range(len(pred)):
        pred[i][..., 2:4] =  arc_sigmoid(torch.sqrt(pred[i][..., 2:4] /
                                   torch_model.model[-1].anchor_grid[i]) / 2)
    return pred


def transform_conf(pred):
    for i in range(len(pred)):
        pred[i][..., 4:] = arc_sigmoid(pred[i][..., 4:])
    return pred


def custom_loss(pred, gt, imgs):
    for i in range(gt.shape[0]):
        gt[i, ..., 0] = i
    torch_gt = torch.from_numpy(gt[gt[..., 1] != nc+1, :])
    strides = torch_model.stride.numpy().astype(int)
    anchor_grid = torch_model.model[-1].anchor_grid
    anchor_num = anchor_grid.shape[2]
    anchor_sizes = (CONFIG['IMGSZ']/strides).astype(int)
    bbox_indices = np.cumsum([0, *[int(anchor_num*(anchor_sizes[i]**2)) for i in range(len(strides))]])
    pt_pred = torch.from_numpy(pred.numpy())
    results = [torch.reshape(pt_pred[:,bbox_indices[i]:bbox_indices[i+1],:],
                                           (-1, anchor_num, anchor_sizes[i], anchor_sizes[i], nc+5))
               for i in range(anchor_grid.shape[0])]
    results = transform_conf(transform_wh(transform_xy(results, strides)))
    res = compute_loss_ota(results, torch_gt, torch.from_numpy(imgs).permute((0, 3, 1, 2)))
    return res[0].numpy()


def pred_visualizer(pred, img):
    out = non_max_suppression(torch.from_numpy(pred[None, ...].numpy()),
                              conf_thres=CONFIG['NMS']['CONF_THRESH'],
                              iou_thres=CONFIG['NMS']['IOU_THRESH'], multi_label=True)[0].numpy()
    out[:, :4] = out[:, :4] / CONFIG['IMGSZ']
    res = bb_array_to_object(out, iscornercoded=True, bg_label=nc+1, is_gt=False)
    return LeapImageWithBBox((img * 255).astype(np.uint8), res)


def gt_visualizer(gt, img):
    gt_permuted = np.concatenate([gt[:, 2:], gt[:, :2]], axis=-1)
    res = bb_array_to_object(gt_permuted, iscornercoded=False, bg_label=nc+1, is_gt=True)
    return LeapImageWithBBox((img * 255).astype(np.uint8), res)


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
def metadata_label(idx: int, preprocess: PreprocessResponse) -> int:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return digit_int


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_encoder, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='classes')
leap_binder.add_prediction(name='classes', labels=['X', 'Y', 'W', ' H', ' Conf'] + data_dict['names'])
leap_binder.set_visualizer(gt_visualizer, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(pred_visualizer, 'bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.add_custom_loss(custom_loss, 'od_loss')
