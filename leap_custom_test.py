from leap_binder import preprocess_func, input_encoder, gt_encoder, custom_loss, pred_visualizer, gt_visualizer
import tensorflow as tf
import numpy as np
from utils.general import non_max_suppression
from leap_utils import bb_array_to_object, draw_image_with_boxes
import torch
from code_loader.contract.visualizer_classes import LeapImageWithBBox
import matplotlib
matplotlib.use('Tkagg')


def check_integration():
    model = tf.keras.models.load_model("yolov7-2.h5")
    res = preprocess_func()
    train = res[0]
    inpt = input_encoder(0, train)[None, ...]
    pred = model(inpt)
    rotated_pred = tf.transpose(pred, (0,2,1))
    gt = gt_encoder(0, train)[None, ...]
    batch_pred = tf.concat([rotated_pred, rotated_pred], axis=0)
    batch_gt = np.concatenate([gt, gt], axis=0)
    imgs = np.concatenate([inpt, inpt], axis=0)
    custom_loss(batch_pred, batch_gt, imgs)
    img_with_bbox = pred_visualizer(rotated_pred[0, ...], imgs[0, ...])
    draw_image_with_boxes(img_with_bbox.data, img_with_bbox.bounding_boxes)
    gt_img_with_bbox = gt_visualizer(gt[0, ...], imgs[0, ...])
    draw_image_with_boxes(gt_img_with_bbox.data, gt_img_with_bbox.bounding_boxes)


if __name__ == '__main__':
    check_integration()
