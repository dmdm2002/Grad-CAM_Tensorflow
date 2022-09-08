import os
import glob
import re
import numpy as np
import tensorflow as tf

from options import params


class Loader(params):
    def __init__(self):
        super(Loader, self).__init__()

        #original_image
        # self.original_img = glob.glob(f'{re.compile("Proposed").sub("CycleGAN", self.root)}/1-fold/B/fake/*')
        # a = re.compile("Proposed").sub("CycleCAN", self.root)
        # print(a)

        # get images path
        self.A_iris = glob.glob(f'{self.A}/iris/*/*')
        self.A_iris_upper = glob.glob(f'{self.A}/iris_upper/*/*')
        self.A_iris_lower = glob.glob(f'{self.A}/iris_lower/*/*')

        self.B_iris = glob.glob(f'{self.B}/iris/*/*')
        self.B_iris_upper = glob.glob(f'{self.B}/iris_upper/*/*')
        self.B_iris_lower = glob.glob(f'{self.B}/iris_lower/*/*')

        # get label
        self.A_label = [self.classes[i.split("\\")[-2]] for i in self.A_iris]
        self.B_label = [self.classes[i.split("\\")[-2]] for i in self.B_iris]

        self.A_label_ds = tf.data.Dataset.from_tensor_slices(tf.one_hot(self.A_label, 2))
        self.B_label_ds = tf.data.Dataset.from_tensor_slices(tf.one_hot(self.B_label, 2))

        # list to ds
        self.A_iris_ds = tf.data.Dataset.from_tensor_slices(self.A_iris)
        self.A_iris_upper_ds = tf.data.Dataset.from_tensor_slices(self.A_iris_upper)
        self.A_iris_lower_ds = tf.data.Dataset.from_tensor_slices(self.A_iris_lower)

        self.B_iris_ds = tf.data.Dataset.from_tensor_slices(self.B_iris)
        self.B_iris_upper_ds = tf.data.Dataset.from_tensor_slices(self.B_iris_upper)
        self.B_iris_lower_ds = tf.data.Dataset.from_tensor_slices(self.B_iris_lower)

    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, 3)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224]) / 255.

        return img

    def load(self):
        A_iris_ds = self.A_iris_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        A_iris_upper_ds = self.A_iris_upper_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        A_iris_lower_ds = self.A_iris_lower_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        A_ds = tf.data.Dataset.zip((A_iris_ds, A_iris_upper_ds, A_iris_lower_ds, self.A_label_ds))
        B_ds = tf.data.Dataset.zip((A_iris_ds, A_iris_upper_ds, A_iris_lower_ds, self.A_label_ds))

        return A_ds, B_ds, self.B_iris

    def configure_for_performance(self, ds, cnt, shuffle=False):
        if shuffle==True:
            ds = ds.shuffle(buffer_size=cnt)
            ds = ds.batch(self.batchsz)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        elif shuffle==False:
            ds = ds.batch(self.batchsz)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds