import sys
sys.path.append('../')
from generators.common import Generator
import os
import numpy as np
import cv2
import tensorflow as tf
import IPython.display as display
tf.compat.v1.disable_eager_execution()

filenames='/home/minjun/Jupyter/Ship_Detection/Data/train_tfrecorder/train_data.tfrecords'
ship_classes = {
    'container': 0,
    'oil tanker': 1,
    'aircraft carrier': 2,
    'maritime vessels': 3
}

image_feature_description  = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/x1': tf.io.VarLenFeature(tf.float32),
    'image/object/y1': tf.io.VarLenFeature(tf.float32),
    'image/object/x2': tf.io.VarLenFeature(tf.float32),
    'image/object/y2': tf.io.VarLenFeature(tf.float32),
    'image/object/x3': tf.io.VarLenFeature(tf.float32),
    'image/object/y3': tf.io.VarLenFeature(tf.float32),
    'image/object/x4': tf.io.VarLenFeature(tf.float32),
    'image/object/y4': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

example = tf.train.Example()
def _features (record):    
    example.ParseFromString(record)
    f = example.features.feature
    return dict(f.items())

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

class ShipGenerator(Generator):
    def __init__(
            self,
            set_name,
            data_dir,
            gen_type = 'train',
            ratio = 1.0,
            classes = ship_classes,
            **kwargs
    ):
        """

        Args:
            data_dir: the path of directory which contains ImageSets directory
            set_name: test|trainval|train|val
            classes: class names tos id mapping
            image_extension: image filename ext
            
            **kwargs:
        """
        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        self.mylist = np.array(np.array(list(map(_features,tf.compat.v1.io.tf_record_iterator(data_dir)))))
        self.length = self.mylist.shape[0]
        if gen_type == 'train' :
            self.mylist= self.mylist[:int(ratio*self.length)]
        elif gen_type == 'val' : 
            self.mylist= self.mylist[int(-ratio*self.length):]
        self.length = self.mylist.shape[0]
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        super(ShipGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return self.length

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        
        h, w = self.mylist[image_index]['image/height'].numpy() ,self.mylist[image_index]['image/width'].numpy()
        return float(w) / float(h)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        encoded_image = self.mylist[image_index]['image/encoded'].bytes_list.value[0]
        image = tf.io.decode_image(encoded_image).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        image_index = image_index[0] if type(image_index)==list else image_index
        length = len(self.mylist[image_index]['image/object/class/label'].int64_list.value) 
        annotations = {'labels': np.empty((length,), dtype=np.int32),
                       'bboxes': np.empty((length, 4), dtype=np.float32),
                       'quadrangles': np.empty((length, 4, 2), dtype=np.float32),
                       }
        for annot in self.mylist[image_index]:
            if self.detect_quadrangle: 
                width, height = self.mylist[image_index]['image/width'].int64_list.value[0],self.mylist[image_index]['image/height'].int64_list.value[0], 

                x1 = np.clip(self.mylist[image_index]['image/object/x1'].float_list.value,0,1)*width
                y1 = np.clip(self.mylist[image_index]['image/object/y1'].float_list.value,0,1)*height
                x2 = np.clip(self.mylist[image_index]['image/object/x2'].float_list.value,0,1)*width
                y2 = np.clip(self.mylist[image_index]['image/object/y2'].float_list.value,0,1)*height
                x3 = np.clip(self.mylist[image_index]['image/object/x3'].float_list.value,0,1)*width
                y3 = np.clip(self.mylist[image_index]['image/object/y3'].float_list.value,0,1)*height
                x4 = np.clip(self.mylist[image_index]['image/object/x4'].float_list.value,0,1)*width
                y4 = np.clip(self.mylist[image_index]['image/object/y4'].float_list.value,0,1)*height
                
                quadrangle = np.array([x1,y1,x2,y2,x3,y3,x4,y4])
                quadrangle = np.transpose(quadrangle) 
                quadrangle = np.reshape(quadrangle,(-1,4,2))
                ordered_quadrangle = self.reorder_vertexes(quadrangle)
                annotations['quadrangles'] = ordered_quadrangle
                annotations['bboxes'] = np.array([[min(_x1,_x2,_x3,_x4),min(_y1,_y2,_y3,_y4),
                                          max(_x1,_x2,_x3,_x4),max(_y1,_y2,_y3,_y4)]
                                         for _x1,_x2,_x3,_x4,_y1,_y2,_y3,_y4 in zip(x1,x2,x3,x4,y1,y2,y3,y4)]) 
                annotations['labels'] = np.array(self.mylist[image_index]['image/object/class/label'].int64_list.value)-1
                
        return annotations
    
    def reorder_vertexes(self, bboxes):
        """
        reorder vertexes as the paper shows, (top, right, bottom, left)
        Args:
            bboxes:

        Returns:

        """
        assert bboxes.shape[1:] == (4, 2)
        ordered_vertexes = np.zeros((len(bboxes),4,2),dtype=np.float32)
        for idx,vertexes in enumerate(bboxes) : 
            xmin, ymin = np.min(vertexes, axis=0)
            xmax, ymax = np.max(vertexes, axis=0)

            # determine the first point with the smallest y,
            # if two vertexes has same y, choose that with smaller x,
            ordered_idxes = np.argsort(vertexes, axis=0)
            ymin1_idx = ordered_idxes[0, 1]
            ymin2_idx = ordered_idxes[1, 1]
            if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
                if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
                    first_vertex_idx = ymin1_idx
                else:
                    first_vertex_idx = ymin2_idx
            else:
                first_vertex_idx = ymin1_idx
            ordered_idxes = [(first_vertex_idx + i) % 4 for i in range(4)]
            ordered_vertexes[idx] = vertexes[ordered_idxes]
            # drag the point to the corresponding edge
            ordered_vertexes[idx,0,1] = ymin
            ordered_vertexes[idx,1,0] = xmax
            ordered_vertexes[idx,2,1] = ymax
            ordered_vertexes[idx,3,0] = xmin
        return ordered_vertexes

