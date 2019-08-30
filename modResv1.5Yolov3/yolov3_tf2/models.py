""" change the yolo_anchors values here based on kmeans output
TO-DO: call kmeans.py from within this file
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    Cropping2D,
    Dropout,
    Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .batch_norm import BatchNormalization
from .utils import broadcast_iou

yolo_anchors = np.array([(103, 132), (120, 225), (142, 166), (153, 274), (178, 192),(180, 352), (197, 266), (243, 337), (245, 216)],np.float32) 

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def Conv2D_BN_Relu(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=False, 
               kernel_regularizer=l2(1e-4))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x

def identity_block(input_tensor, filters, kernel_size, strides=1):
    """ shortcut without conv layer
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
        
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    
    x = Conv2D(filters1, (1,1), padding="same")(input_tensor)
   
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY,
                           epsilon=BATCH_NORM_EPSILON)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY,
                           epsilon=BATCH_NORM_EPSILON)(x)

    x = Add()([x, input_tensor])
    x = LeakyReLU(alpha=0.1)(x)
    
    return x

def conv_block(input_tensor, filters, kernel_size, strides=1):
    """conv_block is the block that has a conv layer at shortcut
    
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    
    filters1, filters2, filters3 = filters
    bn_axis = 3
   
    x = Conv2D(filters1, (1, 1), strides=strides, padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(shortcut)

    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    
    return x

def pool_block(input_tensor, filters, kernel_size, padding_size=(1,1)):
    """ pool_layer block uses MaxPool in shortcut layer
    
    
    """
    
    filters1, filters2, filters3 = filters
    bn_axis = 3

    x = Conv2D(filters1, (1, 1), padding='same', strides=(1,1))(input_tensor)
    
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(x)
    
    x = LeakyReLU(alpha=0.1)(x)
    
    x = ZeroPadding2D(padding=padding_size)(x)

    x = Conv2D(filters2, kernel_size, padding='valid', strides=(2,2))(x)
    
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(x)
    
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters3, (1, 1), padding='same', strides=(1,1))(x)
    x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, 
                           epsilon=BATCH_NORM_EPSILON)(x)
    ## Maxpooling here
    shortcut = MaxPooling2D((1,1), strides=(2, 2), padding='same')(input_tensor)
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    
    return x


def Resnet(name=None):
    x = inputs = Input([None, None, 3])
    bn_axis = 3

    x = Conv2D_BN_Relu(x, 64,(7,7), strides=1)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = conv_block(x, [64, 64, 256], 3, strides=(1, 1))
    x = identity_block(x, [64, 64, 256], 3)
    x = pool_block(x, [64, 64, 256], 3, padding_size=(1,1))
    
    x = conv_block(x, [128, 128, 512], 3)
    x = identity_block(x, [128, 128, 512], 3)
    x = identity_block(x, [128, 128, 512], 3)
    x = r1 = pool_block(x, [128, 128, 512], 3, padding_size=(1,1))
    
    x = conv_block(x, [256, 256, 1024], 3)
    x = identity_block(x, [256, 256, 1024], 3)
    x = identity_block(x, [256, 256, 1024], 3)
    x = identity_block(x, [256, 256, 1024], 3)
    x = identity_block(x, [256, 256, 1024], 3)
    x = r2 = pool_block(x, [256, 256, 1024], 3, padding_size=(1,1))
    
    x = conv_block(x, [512, 512, 2048], 3)
    x = identity_block(x, [512, 512, 2048], 3)
    x = pool_block(x, [512, 512, 2048], 3, padding_size=(1,1))
    return tf.keras.Model(inputs, (r1, r2, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = Conv2D_BN_Relu(x, filters, 1)
            x = UpSampling2D(2)(x)
            if (x.shape[1] != x_skip.shape[1] and x.shape[2] != x_skip.shape[2]):
                x = Cropping2D(cropping=((1,0), (1,0)), input_shape = x.shape[1:])(x)
            elif (x.shape[1] != x_skip.shape[1] and x.shape[2] == x_skip.shape[2]):
                x = Cropping2D(cropping=((1,0), (0,0)), input_shape = x.shape[1:])(x)
            elif (x.shape[1] == x_skip.shape[1] and x.shape[2] != x_skip.shape[2]):
                x = Cropping2D(cropping=((0,0), (1,0)), input_shape = x.shape[1:])(x)
                
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = Conv2D_BN_Relu(x, filters, 1)
        x = Conv2D_BN_Relu(x, filters * 2, 3)
        x = Conv2D_BN_Relu(x, filters, 1)
        x = Conv2D_BN_Relu(x, filters * 2, 3)
        x = Conv2D_BN_Relu(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv



def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Conv2D_BN_Relu(x, filters * 2, 3)
        x = Conv2D_BN_Relu(x,  anchors * (classes + 5),1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = [tf.shape(pred)[1], tf.shape(pred)[2]]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2,2, 1, classes), axis=-1)

#     box_xy = tf.concat((box_x, box_y), axis=-1)
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid[0], tf.float32)) / \
        tf.cast(grid_size, tf.float32)
#     box_y = (box_y + tf.cast(grid[1], tf.float32)) / \
#         tf.cast(grid_size[1], tf.float32)
    box_wh = tf.exp(box_wh) * anchors
    tf.print(box_xy, box_wh)
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=[None, None], channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=2, training=False):
    x = inputs = Input([size[0], size[1], channels])
    

    x_36, x_61, x = Resnet(name='yolo_resnet')(x)

    x = YoloConv(64, name='yolo_conv_0')(x)
    output_0 = YoloOutput(64, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(32, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(32, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(16, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(16, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')



def YoloLoss(anchors, classes, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = [tf.shape(y_true)[1], tf.shape(y_true)[2]]
#         grid_size = [46, 23]
        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
#         true_xy[1] = true_xy[1] * tf.cast(grid_size[1], tf.float32) - \
#             tf.cast(grid[1], tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        
        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
