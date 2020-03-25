import argparse
from datetime import date
import os
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']='6'

# import keras
# import keras.preprocessing.image
# import keras.backend as K
# from keras.optimizers import Adam, SGD

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD

from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet
from losses import smooth_l1, focal, smooth_l1_quad
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from utils.anchors import AnchorParameters

from generators.ship import ShipGenerator
from eval.common import evaluate
import numpy as np


    

def main():
    phi = 1
    weighted_bifpn = False
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    score_threshold = 0.9
    nms_threshold = 0.5
    
    common_args = {
    'batch_size': 1,
    'phi': phi,
    'detect_ship': True,
    'detect_quadrangle': True,
}
    
    #ship_path = '/home/minjun/Jupyter/Ship_Detection/Data/train_tfrecorder/train_data2.tfrecords'
    val_dir = '/home/minjun/Jupyter/Ship_Detection/Data/tfrecorder/val_data_1280.tfrecords'
    model_path = 'checkpoints/reanchor/ship_99_0.3717_0.3662.h5'
    print(model_path)
    

    
#    train_generator = ShipGenerator(
#        'train/ship_detection',
#        ship_path,
#        gen_type='train',
#        ratio = ratio,
#        group_method='none',
#        **common_args
#    )

    validation_generator = ShipGenerator(
        'val/ship_detection',
        val_dir,
        gen_type='val',
        ratio = 1,
        shuffle_groups=False,
        selection=False,
        **common_args
    )
    num_classes = validation_generator.num_classes()
    num_anchors = validation_generator.num_anchors
    anchor_parameters=AnchorParameters.ship
    
    model, prediction_model = efficientdet(phi,
                                           num_classes=num_classes,
                                               num_anchors=num_anchors,
                                           freeze_bn=True,
                                           detect_quadrangle=True,
                                           anchor_parameters=anchor_parameters,
                                           score_threshold=score_threshold,
                                           nms_threshold=0.7
                                           )
    prediction_model.load_weights(model_path, by_name=True)
    
#    print(evaluate(generator=train_generator,
#             model = prediction_model,
#             score_threshold=0.01,
#             max_detections=100,
#             visualize=False,
#            )
#         )
    if False:
        for i in np.arange(0.2, 1, 0.05):
            print(evaluate(generator=validation_generator,
                     model = prediction_model,
                     score_threshold=score_threshold,
                     max_detections=300,
                     visualize=False,
                     nms_threshold=i,
                    )
                 )
    print(evaluate(generator=validation_generator,
                     model = prediction_model,
                     score_threshold=score_threshold,
                     max_detections=300,
                     visualize=True,
                     nms_threshold=nms_threshold,
                    )
         )
    #colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    
# score_threshold = 0.01
#num_fp=2439.0, num_tp=14580.0
#{0: (0.859279664856701, 4106.0), 1: (0.8278047641932937, 1579.0), 2: (0.40426303380023887, 57.0), 3: (0.8525899236151595, 10989.0)}


#num_fp=2439.0, num_tp=14580.0
#{0: (0.859279664856701, 4106.0), 1: (0.8278047641932937, 1579.0), 2: (0.40426303380023887, 57.0), 3: (0.8525899236151595, 10989.0)}

#num_fp=1186.0, num_tp=20099.0 , selection
# {0: (0.8863557966508845, 2413.0), 1: (0.8712331102638979, 1579.0), 2: (1.0, 3.0), 3: (0.8585311515528976, 19048.0)}


if __name__ == '__main__':
    main()
