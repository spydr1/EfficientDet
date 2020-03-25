#-*- coding: future_fstrings -*-
from model import efficientdet
import cv2
import os
import numpy as np
import time
from utils import preprocess_image
from utils.anchors import anchors_for_shape, AnchorParameters
import os.path as osp
from generators.ship import ShipGenerator
from tensorflow import keras
from utils.anchors import AnchorParameters

import glob
import argparse
from tqdm import tqdm
import tensorflow as tf
import csv
import re
import pdb

phi = 1
weighted_bifpn = False
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]
classes_name = ['container','oil tanker','aircraft carrier','maritime vessels']
num_classes = len(classes_name)
AnchorParameters.ship = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # ratio=h/w
    ratios=np.array([0.25,0.5,2,4], keras.backend.floatx()),
    scales=np.array([0.25, 0.5, 0.75, 1.0], keras.backend.floatx()),
)
colors = [np.random.randint(100, 256, 3).tolist() for i in range(num_classes)]

from shapely.geometry import Polygon

def nms (quadrangles,boxes,scores,classes,ratios, threshold=0.7):
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    quadrangles = quadrangles[order]
    boxes = boxes[order]
    classes = classes[order]
    ratios = ratios[order]
    keep = [True]*len(order)

    for i,p1 in enumerate(quadrangles):
        p1 = Polygon(((p1[0],p1[1]),
                         (p1[2],p1[3]),
                         (p1[4],p1[5]),
                         (p1[6],p1[7])))
        for j,p2 in enumerate(quadrangles):

            p2 = Polygon(((p2[0],p2[1]),
                         (p2[2],p2[3]),
                         (p2[4],p2[5]),
                         (p2[6],p2[7])))
            
            inter = p1.intersection(p2).area
            union = p1.area+p2.area-inter
            iou = inter/(union+1e-5)
            if j>i and iou>threshold:
                keep[j]=False
    quadrangles = quadrangles[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    ratios = ratios[keep]
    return quadrangles,boxes,scores,classes,ratios

def save_det_to_csv(dst_path, det_by_file):
    """ Save detected objects to CSV format

    :param str dst_path: Path to save csv
    :param dict det_by_file: detected objects that key is filename
    :return: None (save csv file)
    """
    with open(dst_path, 'w') as f:
        w = csv.DictWriter(f, ['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y',
                               'point3_x', 'point3_y', 'point4_x', 'point4_y'])
        w.writeheader()

        for file_path, det in det_by_file.items():
            boxes = det['boxes']
            classes = det['classes']
            scores = det['scores']

            for box, cls, score in zip(boxes, classes, scores):
                det_dict = {'file_name': os.path.basename(file_path),
                            'class_id': cls+1,
                            'confidence': score,
                            'point1_x': box[0],
                            'point1_y': box[1],
                            'point2_x': box[2],
                            'point2_y': box[3],
                            'point3_x': box[4],
                            'point3_y': box[5],
                            'point4_x': box[6],
                            'point4_y': box[7],
                            }
                w.writerow(det_dict)

def get_patch_generator(image, patch_size, overlay_size):
    """ Patch Generator to split image by grid

    :param numpy image: source image
    :param int patch_size: patch size that width and height of patch is equal
    :param overlay_size: overlay size in patches
    :return: generator for patch image, row and col coordinates
    """
    step = patch_size - overlay_size
    for row in range(0, image.shape[0] - overlay_size, step):
        for col in range(0, image.shape[1] - overlay_size, step):
            # Handling for out of bounds
            patch_image_height = patch_size if image.shape[0] - row > patch_size else image.shape[0] - row
            patch_image_width = patch_size if image.shape[1] - col > patch_size else image.shape[1] - col

            # Set patch image
            patch_image = image[row: row + patch_image_height, col: col + patch_image_width]

            # Zero padding if patch image is smaller than patch size
            if patch_image_height < patch_size or patch_image_width < patch_size:
                pad_height = patch_size - patch_image_height
                pad_width = patch_size - patch_image_width
                patch_image = np.pad(patch_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

            yield patch_image, row, col

def inference(model_path, image_dir, dst_path, patch_size, overlay_size, save_img, test_one,score_threshold,nms_threshold,model_nms_threshold):
    """ Inference images to detect objects

    :param str ckpt_path: path to trained checkpoint
    :param str image_dir: directory to source images
    :param str dst_path: path to save detection output
    :param int patch_size: patch size that width and height of patch is equal
    :param int overlay_size: overlay size in patches
    :return: None (save detection output)

    """
    # Get filenames
    file_paths = [os.path.join(root, name) for root, dirs, files in os.walk(image_dir) for name in files if
                  name.endswith('png') or name.endswith('jpg')]
    
    model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       num_anchors=AnchorParameters.ship.num_anchors(),
                                       score_threshold=score_threshold,
                                       detect_quadrangle=True,
                                       anchor_parameters=AnchorParameters.ship,
                                       nms_threshold = model_nms_threshold)
    print(model_path)
    prediction_model.load_weights(model_path, by_name=True)
    det_by_file = dict()
    
    patch_size = args.patch_size
    overlay_size = args.overlay_size
    if test_one :
        file_paths = file_paths[:20]
    
    for file_path in tqdm(file_paths):
        start = time.time() 
        image = cv2.imread(file_path)
        src_image = image
        patch_generator = get_patch_generator(image, patch_size=patch_size, overlay_size=overlay_size)

        classes_list, scores_list, quadrangles_list, boxes_list,ratios_list = list(), list(), list(), list(), list()
        
        for patch_image, row, col in patch_generator:
            #print("row {} col {}".format(row,col))
            image, scale, offset_h, offset_w = preprocess_image(patch_image, image_size=image_size)
            inputs = np.expand_dims(image, axis=0)
            anchors = anchors_for_shape((image_size, image_size), anchor_params=AnchorParameters.ship)
            # run network
            boxes, scores, alphas, ratios, classes = prediction_model.predict([np.expand_dims(image, axis=0),
                                                                                       np.expand_dims(anchors, axis=0)])
            h, w = patch_image.shape[:2]
            
            alphas = 1 / (1 + np.exp(-alphas))
            ratios = 1 / (1 + np.exp(-ratios))
            quadrangles = np.zeros(boxes.shape[:2] + (8,))
            quadrangles[:, :, 0] = boxes[:, :, 0] + (boxes[:, :, 2] - boxes[:, :, 0]) * alphas[:, :, 0]
            quadrangles[:, :, 1] = boxes[:, :, 1]
            quadrangles[:, :, 2] = boxes[:, :, 2]
            quadrangles[:, :, 3] = boxes[:, :, 1] + (boxes[:, :, 3] - boxes[:, :, 1]) * alphas[:, :, 1]
            quadrangles[:, :, 4] = boxes[:, :, 2] - (boxes[:, :, 2] - boxes[:, :, 0]) * alphas[:, :, 2]
            quadrangles[:, :, 5] = boxes[:, :, 3]
            quadrangles[:, :, 6] = boxes[:, :, 0]
            quadrangles[:, :, 7] = boxes[:, :, 3] - (boxes[:, :, 3] - boxes[:, :, 1]) * alphas[:, :, 3]

            boxes[0, :, [0, 2]] = boxes[0, :, [0, 2]] - offset_w
            boxes[0, :, [1, 3]] = boxes[0, :, [1, 3]] - offset_h
            boxes /= scale
            boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1) + col
            boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1) + row
            boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1) + col
            boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1) + row

            quadrangles[0, :, [0, 2, 4, 6]] = quadrangles[0, :, [0, 2, 4, 6]] - offset_w
            quadrangles[0, :, [1, 3, 5, 7]] = quadrangles[0, :, [1, 3, 5, 7]] - offset_h
            quadrangles /= scale
            quadrangles[0, :, [0, 2, 4, 6]] = np.clip(quadrangles[0, :, [0, 2, 4, 6]], 0, w - 1) + col
            quadrangles[0, :, [1, 3, 5, 7]] = np.clip(quadrangles[0, :, [1, 3, 5, 7]], 0, h - 1) + row
            
            #[1, 3, 5, 7]]
            #[0, 2, 4, 6]
            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > score_threshold)[0]

            # select those detections
            boxes = boxes[0, indices]
            scores = scores[0, indices]
            classes = classes[0, indices]
            quadrangles = quadrangles[0, indices]
            ratios = ratios[0, indices]
                            
            #quadrangles = np.array(quadrangles).reshape(-1,8)
            #boxes = np.array(boxes_list).reshape(-1, 4)
            
            if len(quadrangles)>0 :
                quadrangles_list.extend(list(quadrangles))
                boxes_list.extend(list(boxes))
                classes_list.extend(list(classes))
                scores_list.extend(list(scores))
                ratios_list.extend(list(ratios))
        quadrangles = np.array(quadrangles_list).reshape(-1, 8)
        boxes = np.array(boxes_list).reshape(-1, 4)
        classes = np.array(classes_list).flatten()
        scores = np.array(scores_list).flatten()
        ratios = np.array(ratios_list).flatten()
        #quadrangles = quadrangles[scores > 0]
        #classes = classes[scores > 0]
        #scores = scores[scores > 0]
        #pdb.set_trace()
        quadrangles, boxes, classes, scores,ratios = nms(quadrangles, boxes, classes, scores, ratios , nms_threshold)
        det_by_file[file_path] = {'boxes': quadrangles, 'classes': classes, 'scores': scores}
        #print(time.time() - start)

    # Save detection output
        if save_img:
            for bbox, score, label, quadrangle, ratio in zip(boxes, scores, classes, quadrangles, ratios):
                xmin = int(round(bbox[0]))
                ymin = int(round(bbox[1]))
                xmax = int(round(bbox[2]))
                ymax = int(round(bbox[3]))
                
                score = '{:.4f}'.format(score)
                class_id = int(label)
                color = colors[class_id]
                class_name = classes_name[class_id]
                label = '-'.join([class_name, score])
                ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                #cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
                #cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
                #cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                #cv2.putText(src_image, score, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                #cv2.putText(src_image, f'{ratio:.2f}', (xmin + (xmax - xmin) // 3, (ymin + ymax) // 2),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.drawContours(src_image, [quadrangle.astype(np.int32).reshape((4, 2))], -1, color, 3)
            cv2.imwrite(dst_path+'/img/ship{}.jpg'.format(int(re.findall("\d+",file_path)[0])),src_image)
        #if test_one :
        #    break
            
    save_det_to_csv(dst_path+'/result.csv', det_by_file)


#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image', src_image)
#cv2.waitKey(0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str,
                        help='Path to trained checkpoint, typically of the form path/to/model-%step.ckpt')
    parser.add_argument('--image_dir', type=str,
                        help='Path to images to be inferred')
    parser.add_argument('--dst_path', type=str,
                        help='Path to save detection output')
    parser.add_argument('--patch_size', type=int, default=1024,
                        help='Patch size, width and height of patch is equal.')
    parser.add_argument('--overlay_size', type=int, default=384,
                        help='Overlay size for patching.')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='score_threshold')
    parser.add_argument('--model_nms_threshold', type=float, default=0.7,
                    help='model_nms_threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='nms_threshold')
    parser.add_argument('--save_img', help='does save the image?', action='store_true',default=False)
    parser.add_argument('--test_one', help='just one image?', action='store_true',default=False)

    args = parser.parse_args()

    inference(**vars(args))
