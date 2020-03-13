import argparse
from datetime import date
import os
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

common_args = {
    'batch_size': 1,
    'phi': 1,
    'detect_ship': True,
    'detect_quadrangle': True,
}

    

def main():
    phi = 1
    weighted_bifpn = False
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    score_threshold = 0.5
    
    ship_path = '/home/minjun/Jupyter/Ship_Detection/Data/train_tfrecorder/train_data2.tfrecords'
    model_path = 'checkpoints/2020-03-10/ship_08_1.3620_2.1948.h5'
    ratio = 0.8
    
    train_generator = ShipGenerator(
        'train/ship_detection',
        ship_path,
        gen_type='train',
        ratio = ratio,
        group_method='none',
        **common_args
    )

    validation_generator = ShipGenerator(
        'val/ship_detection',
        ship_path,
        gen_type='val',
        ratio = 1-ratio,
        shuffle_groups=False,
        **common_args
    )
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors
    anchor_parameters=AnchorParameters.ship
    
    model, prediction_model = efficientdet(1,
                                           num_classes=num_classes,
                                           num_anchors=num_anchors,
                                           freeze_bn=True,
                                           detect_quadrangle=True,
                                           anchor_parameters=anchor_parameters,
                                           score_threshold=score_threshold 
                                           )
    prediction_model.load_weights(model_path, by_name=True)
    
    print(evaluate(generator=train_generator,
             model = prediction_model,
             iou_threshold=0.5,
             score_threshold=0.01,
             max_detections=100,
             visualize=False,
            )
         )
          
    print(evaluate(generator=validation_generator,
             model = prediction_model,
             iou_threshold=0.5,
             score_threshold=0.01,
             max_detections=100,
             visualize=False,
            )
         )
    #colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]

if __name__ == '__main__':
    main()
