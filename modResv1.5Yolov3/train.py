""" train file with different image height and width"""

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import pandas as pd
import cv2, os
import re
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloLoss,
    yolo_anchors, yolo_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset 

flags.DEFINE_string('train_dataset', 'train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', 'val.tfrecord', 'path to validation dataset')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', 'class.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('height', 1088, 'image height')
flags.DEFINE_integer('width', 1088, 'image width')
flags.DEFINE_integer('epochs', 100, 'number of epochs')
flags.DEFINE_integer('batch_size', 1, 'batch size per GPU')
flags.DEFINE_integer('num_gpus', 2, 'number of GPUs')
flags.DEFINE_integer('class_num', 2, 'number of classes')


size = (FLAGS.height, FLAGS.width)
GPU_BATCH_SIZE = FLAGS.num_gpus * FLAGS.batch_size
EPOCHS = FLAGS.epochs
lr = 0.5
class_num = FLAGS.class_num
        
def main(_argv):
    if FLAGS.num_gpus == 0:
        distribution = tf.distribute.OneDeviceStrategy('device:CPU:0')
    elif FLAGS.num_gpus == 1:
        distribution = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
        distribution = tf.distribute.MirroredStrategy()
    ## MirroredStrategy distributes equally across all the nodes, for other types of strategy, refer (https://www.tensorflow.org/guide/distribute_strategy)
    with distribution.scope():
        model = YoloV3(size, training=True, classes=class_num)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
        if FLAGS.transfer != 'none':
            model.load_weights(FLAGS.weights)
            if FLAGS.transfer == 'fine_tune':
                # freeze darknet
                darknet = model.get_layer('yolo_darknet')
                freeze_all(darknet)
            elif FLAGS.mode == 'frozen':
                # freeze everything
                freeze_all(model)
            else:
                # reset top layers
                if FLAGS.tiny:  # get initial weights
                    init_model = YoloV3Tiny(FLAGS.size, training=True)
                else:
                    init_model = YoloV3(FLAGS.size, training=True)
    
                if FLAGS.transfer == 'darknet':
                    for l in model.layers:
                        if l.name != 'yolo_darknet' and l.name.startswith('yolo_'):
                            l.set_weights(init_model.get_layer(
                                l.name).get_weights())
                        else:
                            freeze_all(l)
                elif FLAGS.transfer == 'no_output':
                    for l in model.layers:
                        if l.name.startswith('yolo_output'):
                            l.set_weights(init_model.get_layer(
                                l.name).get_weights())
                        else:
                            freeze_all(l)
        optimizer = tf.keras.optimizers.Adam(lr=0.5)
#         optimizer = AdamAccumulate(lr=FLAGS.learning_rate, accum_iters=gpu_batches_to_accum)
        loss = [YoloLoss(anchors[mask], classes=class_num) for mask in anchor_masks]
        model.compile(optimizer=optimizer, loss=loss,run_eagerly=False)
        
    def input_fn(train_dataset, val_dataset, classes, batch_size):
        train_dataset = dataset.load_tfrecord_dataset(train_dataset, classes, size)
#         train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(lambda x, y: (
            dataset.transform_images(x, size),
            dataset.transform_targets(y, anchors, anchor_masks, 2, size)))
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    
        val_dataset = dataset.load_tfrecord_dataset(val_dataset, classes, size)
        val_dataset = val_dataset.batch(batch_size).repeat(EPOCHS)
        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, size),
            dataset.transform_targets(y, anchors, anchor_masks, 2, size)))
        
        return train_dataset, val_dataset
    
    checkpoint_directory = "checkpoints"
    os.makedirs(checkpoint_directory)
    
    callbacks = [
            ReduceLROnPlateau(monitor='loss',
                               factor=0.2,
                               patience=2,
                               verbose=1,
                               mode='auto',
                               cooldown=0,
                               min_lr=0
                                ),
            ModelCheckpoint(os.path.join(checkpoint_directory,
                                        'yolov3_train_{epoch}.tf'),verbose=1, save_weights_only=True),
            TensorBoard(log_dir='./tensorboard/graph',
                            histogram_freq=0,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None
                        )]        
    

    train_dataset, val_dataset = input_fn(FLAGS.train_dataset, FLAGS.val_dataset, FLAGS.classes, GPU_BATCH_SIZE)
    history = model.fit(train_dataset,epochs=EPOCHS,callbacks=callbacks, validation_data=val_dataset, validation_steps = 2)
    model.save('./resyolo.h5', include_optimizer=True)


if __name__ == '__main__':
    app.run(main)