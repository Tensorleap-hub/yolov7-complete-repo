from typing import List, Union

import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse 
from code_loader.contract.enums import Metric, DatasetMetadataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from utils.datasets import create_dataloader
from leapcfg.config import CONFIG
import torch
import yaml
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    # Hyperparameters
    with open(CONFIG['HYP']) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    with open(CONFIG['DATA']) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
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
    np_gt = np.ones((CONFIG['MAX_INSTANCE'], 6))
    instances_count = torch_gt.shape[0]
    np_gt[:instances_count, :] = torch_gt[:min(CONFIG['MAX_INSTANCE'], instances_count), :]
    return np_gt

def custom_loss(gt, pred):
    torch_gt = torch.from_numpy(gt[gt[...,0] == 0, :])
# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
def metadata_label(idx: int, preprocess: PreprocessResponse) -> int:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return digit_int


def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    return LeapHorizontalBar(data, LABELS)


LABELS = ['0','1','2','3','4','5','6','7','8','9']
# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_encoder, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='classes')
# leap_binder.set_metadata(function=metadata_label, metadata_type=DatasetMetadataType.int, name='label')
leap_binder.add_prediction(name='classes', labels=LABELS)
leap_binder.set_visualizer(name='horizontal_bar_classes', function=bar_visualizer, visualizer_type=LeapHorizontalBar.type)
