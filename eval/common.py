"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations

import numpy as np
import cv2
import progressbar

assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

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

def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def _get_detections(generator, model, score_threshold=0.05, max_detections=100, visualize=False,):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        image = generator.load_image(i)
        src_image = image.copy()

        anchors = generator.anchors
        image, scale, offset_h, offset_w = generator.preprocess_image(image)
        h, w = image.shape[:2]

        # run network
        boxes, scores, *_, labels = model.predict([np.expand_dims(image, axis=0),
                                                            np.expand_dims(anchors, axis=0)])
        print(len(boxes[0]))
        boxes[..., [0, 2]] = boxes[..., [0, 2]] - offset_w
        boxes[..., [1, 3]] = boxes[..., [1, 3]] - offset_h
        boxes /= scale
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
        boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
        boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        # (n, 4)
        image_boxes = boxes[0, indices[scores_sort], :]
        # (n, )
        image_scores = scores[scores_sort]
        # (n, )
        image_labels = labels[0, indices[scores_sort]]
        # (n, 6)
        detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if visualize:
            draw_annotations(src_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(src_image, detections[:5, :4], detections[:5, 4], detections[:5, 5].astype(np.int32),
                            label_to_name=generator.label_to_name,
                            score_threshold=score_threshold)

            # cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
            cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
            cv2.imshow('{}'.format(i), src_image)
            cv2.waitKey(0)

        # copy detections to all_detections
        for class_id in range(generator.num_classes()):
            all_detections[i][class_id] = detections[detections[:, -1] == class_id, :-1]

    return all_detections

classes_name = ['container','oil tanker','aircraft carrier','maritime vessels']
def _get_detections_quad(generator, model, score_threshold=0.05, max_detections=100, visualize=False,nms_threshold =0.2, colors =None):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        image = generator.load_image(i)
        h, w = image.shape[:2]
        src_image = image.copy()
        anchors = generator.anchors
        image, scale, offset_h, offset_w = generator.preprocess_image(image)
        # run network
        boxes, scores, alphas, ratios, classes = model.predict([np.expand_dims(image, axis=0),
                                                            np.expand_dims(anchors, axis=0)])
        
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
        boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
        boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
        boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
        boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)

        quadrangles[0, :, [0, 2, 4, 6]] = quadrangles[0, :, [0, 2, 4, 6]] - offset_w
        quadrangles[0, :, [1, 3, 5, 7]] = quadrangles[0, :, [1, 3, 5, 7]] - offset_h
        quadrangles /= scale
        quadrangles[0, :, [0, 2, 4, 6]] = np.clip(quadrangles[0, :, [0, 2, 4, 6]], 0, w - 1) 
        quadrangles[0, :, [1, 3, 5, 7]] = np.clip(quadrangles[0, :, [1, 3, 5, 7]], 0, h - 1) 

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]
        # select those scores
        scores = scores[0][indices]
        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]
        # select detections
        image_boxes = boxes[0, indices[scores_sort]]
        image_scores = scores[scores_sort]
        image_classes = classes[0, indices[scores_sort]]
        image_quadrangles = quadrangles[0, indices[scores_sort]]
        image_ratios = ratios[0, indices[scores_sort]]
        
        
        image_quadrangles, image_boxes, image_classes, image_scores,image_ratios = nms(image_quadrangles, 
                                                                                       image_boxes, 
                                                                                       image_classes, 
                                                                                       image_scores, 
                                                                                       image_ratios, 
                                                                                       nms_threshold)



        detections = np.concatenate(
            [image_quadrangles, np.expand_dims(image_scores, axis=1), np.expand_dims(image_classes, axis=1)], axis=1)


        if visualize:
            for score, label, quadrangle in zip(image_scores, image_classes, image_quadrangles):
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
            cv2.imwrite('val/img/ship{}.jpg'.format(i),src_image)


        # copy detections to all_detections
        for class_id in range(generator.num_classes()):
            all_detections[i][class_id] = detections[detections[:, -1] == class_id, :-1]

    return all_detections


def _get_annotations(generator,quad=True):
    """
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        A list of lists containing the annotations for each image in the generator.

    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            if quad:
                all_annotations[i][label] = annotations['quadrangles'][annotations['labels'] == label, :].copy()

            else:
                all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations

from shapely.geometry import Polygon
def _compute_overlap_quad(boxes,query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for n,p1 in enumerate(boxes):
        p1 = Polygon(((p1[0],p1[1]),
                         (p1[2],p1[3]),
                         (p1[4],p1[5]),
                         (p1[6],p1[7])))
        for k,p2 in enumerate(query_boxes):
            p2 = Polygon(((p2[0]),
                         (p2[1]),
                         (p2[2]),
                         (p2[3])))  
            inter = p1.intersection(p2).area
            union = p1.area+p2.area-inter
            iou = inter/(union+1e-5)
            overlaps[n, k] = iou
    return overlaps


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=100,
        visualize=False,
        epoch=0,
        nms_threshold = 0.2
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.

    Returns:
        A dict mapping class names to mAP scores.

    """
    # gather all detections and annotations
    colors = [np.random.randint(100, 256, 3).tolist() for i in range(generator.num_classes())]
    all_detections = _get_detections_quad(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     visualize=visualize,nms_threshold=nms_threshold, colors=colors)
    all_annotations = _get_annotations(generator,quad=True)
    average_precisions = {}
    num_tp = 0
    num_fp = 0
    tot_ap=[]
    # process detections and annotations
    
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = _compute_overlap_quad(np.expand_dims(d, axis=0).astype(np.double), annotations.astype(np.double))
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                #if max_overlap >= iou_threshold :
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        if false_positives.shape[0] == 0:
            num_fp += 0
        else:
            num_fp += false_positives[-1]
        if true_positives.shape[0] == 0:
            num_tp += 0
        else:
            num_tp += true_positives[-1]

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    print('num_fp={}, num_tp={}'.format(num_fp, num_tp))

    return average_precisions


if __name__ == '__main__':
    from generators.pascal import PascalVocGenerator
    from model import efficientdet
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 1
    weighted_bifpn = False
    common_args = {
        'batch_size': 1,
        'phi': phi,
    }
    test_generator = PascalVocGenerator(
        'datasets/VOC2007',
        'test',
        shuffle_groups=False,
        skip_truncated=False,
        skip_difficult=True,
        **common_args
    )
    model_path = 'checkpoints/2019-12-03/pascal_05_0.6283_1.1975_0.8029.h5'
    input_shape = (test_generator.image_size, test_generator.image_size)
    anchors = test_generator.anchors
    num_classes = test_generator.num_classes()
    model, prediction_model = efficientdet(phi=phi, num_classes=num_classes, weighted_bifpn=weighted_bifpn)
    prediction_model.load_weights(model_path, by_name=True)
    average_precisions = evaluate(test_generator, prediction_model, visualize=False)
    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations), test_generator.label_to_name(label),
              'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('mAP: {:.4f}'.format(mean_ap))
