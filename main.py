import tensorflow as tf
import tensorflow.keras.backend as K
# import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

from DataLoader import Loader
from model import Model
from CAM import GradCAM, overlay_gradCAM
from temp import *


def GradCAM(img_tensor, model, class_index, activation_layer):
    # y_c : class_index에 해당하는 CNN 마지막 LAYER OP(softmax, linear, ...)의 입력
    model_input = model.input
    print(model.outputs[0].op.inputs[0])
    y_c = model.output[0].op.intputs[0][0, class_index]

    A_k = model.get_layer(activation_layer).output

    get_output = K.function([model_input], [A_k, K.gradients(y_c, A_k)[0]])
    [conv_output, grad_val] = get_output([img_tensor])

    conv_output = conv_output[0]
    grad_val = grad_val[0]

    weights = np.mean(grad_val, axis=(0, 1))

    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])

    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, : ,k]

    grad_cam = np.maximum(grad_cam, 0)

    return grad_cam, weights


Acnt = 4554
Bcnt = 5018

load = Loader()
A_ds, B_ds, original_img = load.load()

# temp = original_img[0]
# temp = temp.split("\\")[-1]
# print(temp)

builder = Model()

iris_model = builder.baseModel()
iris_upper_model = builder.baseModel()
iris_lower_model = builder.baseModel()

fusion_model = builder.fusionModel_shufflenet(iris_model, iris_upper_model, iris_lower_model)

fusion_model.summary()
# set optimizer, loss_function, acc_metric
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_mean = tf.keras.metrics.Mean()

fusion_ckp = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=fusion_model)

OUTPUT_DIR = 'Z:/Iris_dataset/GradCAM/nd_alpha2.0/1-fold/B_blur/shuffle_stage2_2/'


def test_step(x, y):
    val_logits = fusion_model(x, training=False)
    loss_val = loss_fn(y, val_logits)
    test_acc_metric.update_state(y, val_logits)
    loss_mean.update_state(loss_val)

    return val_logits


for i in range(46, 47):
    fusion_ckp_path = f"Z:/backup/ckp/Proposed/nd/fusion_model/1-fold/1-Dense-addCNN/ckpt-{i}"
    fusion_ckp.restore(fusion_ckp_path)

    B_ds_shuffle = load.configure_for_performance(B_ds, Bcnt, shuffle=False)
    B_ds_it = iter(B_ds_shuffle)

    # A_ds_shuffle = load.configure_for_performance(A_ds, Acnt, shuffle=False)
    # A_ds_it = iter(A_ds_shuffle)

    for step in range(Bcnt):
        iris_img, iris_uppper_img, iris_lower_img, iris_label = next(B_ds_it)
        inputs = [iris_img, iris_uppper_img, iris_lower_img]
        # cam = generate_grad_cam(inputs, fusion_model, step, 'concat_layer')
        # cam = GradCAM(fusion_model, 're_lu_24')
        val_logits = test_step(inputs, iris_label)

        classIdx = np.argmax(val_logits.numpy())

        # print(classIdx)
        upsample_size = (224, 224)
        # re_lu_16
        grad_cam = make_gradcam_heatmap(img_array=inputs, model=fusion_model, last_conv_layer_name='re_lu_24', pred_index=classIdx)

        temp = re.compile('Proposed').sub('blur', original_img[step])
        temp = re.compile('B_blur').sub('B', temp)
        temp = re.compile('/iris').sub('', temp)

        img = tf.io.read_file(temp)
        img = tf.image.decode_png(img, 3)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])

        grad_img_name = temp.split("\\")[-1]
        cls = temp.split("\\")[-2]

        OUTPUT_DIR_current= f'{OUTPUT_DIR}/{cls}/'
        if not os.path.exists(f'{OUTPUT_DIR_current}'):
            os.makedirs(f'{OUTPUT_DIR_current}')

        save_path = f'{OUTPUT_DIR_current}/{grad_img_name}'

        new_img = save_and_display_gradcam(img, grad_cam, cam_path=save_path)