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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

phi = 5
weighted_bifpn = False
model_path = 'checkpoints/2020-03-10/ship_08_1.3620_2.1948.h5'
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]
classes_name = ['container','oil tanker','aircraft carrier','maritime vessels']
num_classes = len(classes_name)
AnchorParameters.ship = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # ratio=h/w
    ratios=np.array([0.5,1, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)
score_threshold = 0.5
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return np.array(pick)

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

def inference(model_path, image_dir, dst_path, patch_size, overlay_size, save_img, test_one,score_threshold,nms_threshold ):
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
                  name.endswith('png')]
    
    model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       num_anchors=AnchorParameters.ship.num_anchors(),
                                       score_threshold=score_threshold,
                                       detect_quadrangle=True,
                                       anchor_parameters=AnchorParameters.ship,
                                       nms_threshold = nms_threshold)
    prediction_model.load_weights(model_path, by_name=True)
    det_by_file = dict()
    
    patch_size = args.patch_size
    overlay_size = args.overlay_size
    if test_one :
        file_paths = file_paths[:1]
    
    for file_path in tqdm(file_paths):
        start = time.time() 
        image = cv2.imread(file_path)
        src_image = image
        patch_generator = get_patch_generator(image, patch_size=patch_size, overlay_size=overlay_size)

        classes_list, scores_list, quadrangles_list, boxes_list,ratios_list = list(), list(), list(), list(), list()
        
        for patch_image, row, col in patch_generator:
            image, scale, offset_h, offset_w = preprocess_image(patch_image, image_size=image_size)
            inputs = np.expand_dims(image, axis=0)
            anchors = anchors_for_shape((image_size, image_size), anchor_params=AnchorParameters.ship)
            # run network
            boxes, scores, alphas, ratios, classes = prediction_model.predict([np.expand_dims(image, axis=0),
                                                                                       np.expand_dims(anchors, axis=0)])

            
            h, w = image.shape[:2]



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
        
        selected_indices = non_max_suppression_fast(boxes, overlapThresh = nms_threshold)
        #print(selected_indices)
        boxes = boxes[selected_indices]
        quadrangles = quadrangles[selected_indices]
        classes = classes[selected_indices]
        scores =  scores[selected_indices]
        ratios = ratios[selected_indices]
        #print("prediction length :", len(boxes),len(scores),len(classes),len(quadrangles),len(ratios))

        det_by_file[file_path] = {'boxes': quadrangles, 'classes': classes, 'scores': scores}
        print(time.time() - start)

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
                cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
                #cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
                cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                #cv2.putText(src_image, score, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                #cv2.putText(src_image, f'{ratio:.2f}', (xmin + (xmax - xmin) // 3, (ymin + ymax) // 2),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.drawContours(src_image, [quadrangle.astype(np.int32).reshape((4, 2))], -1, color, 1)
            cv2.imwrite('test2/img/ship{}.jpg'.format(int(re.findall("\d+",file_path)[0])),src_image)
        #if test_one :
        #    break
            
    save_det_to_csv(dst_path, det_by_file)


#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image', src_image)
#cv2.waitKey(0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    
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
    parser.add_argument('--score_threshold', type=float, default=0.7,
                        help='score_threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='nms_threshold')
    parser.add_argument('--save_img', help='does save the image?', action='store_true',default=False)
    parser.add_argument('--test_one', help='just one image?', action='store_true',default=False)

    args = parser.parse_args()

    inference(**vars(args))
