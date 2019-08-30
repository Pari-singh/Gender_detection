import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', 'class.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints_fullds/yolov3_train_18.tf',
                    'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', '../Wave_annotation_set3/train/000016_ch02_20190314104956_000158.jpeg', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
size = (1088, 1088)
def main(_argv):
    yolo = YoloV3()

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_jpeg(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, size)
    tf.print(img)

    t1 = time.time()
    boxes, scores, classes, nms = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    tf.print(nms)
    tf.print(boxes, scores, classes, nms)
#     range_nms = nms[0].to_int64
    for i in range(1):
        print("Did I enter? ")
        print("class_name: ", class_names[int(classes[0][i])])
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.imread(FLAGS.image)
    img = draw_outputs(img, (boxes, scores, classes, nms), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))

if __name__ == '__main__':
    app.run(main)
   


# import matplotlib.pyplot as plt
# import time
# from absl import app, flags, logging
# from absl.flags import FLAGS
# import cv2
# import numpy as np
# import tensorflow as tf
# from yolov3_tf2.models import (
#     YoloV3
# )


# from yolov3_tf2.dataset import transform_images
# from yolov3_tf2.utils import draw_outputs



# images_url=["../Wave_annotation_set3/test/000015_ch02_20190314083337_007973.jpeg", "../Wave_annotation_set3/test/000015_ch02_20190314083337_006701.jpeg"]



# flags.DEFINE_string('classes', 'class.names', 'path to classes file')
# flags.DEFINE_string('weights', './tmp3/training_checkpoints/model_10.h5',
#                     'path to weights file')
# flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
# flags.DEFINE_string('output_file', './output.jpg', 'path to output image')

# size = (1920, 1088)
# def main(_argv):
#     del _argv
#     yolo = YoloV3()
# #    yolo.summary()
#     yolo.load_weights(FLAGS.weights)
#     logging.info('weights loaded')

#     class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
#     logging.info('classes loaded')

#     url=images_url.copy()
#     img=url[0]
#     img = tf.image.decode_image(open(img, 'rb').read(), channels=3)
#     img = tf.expand_dims(img, 0)  
#     im= transform_images(img, size)

#     for pic in url[1:]:

#         img1 = tf.image.decode_image(open(pic, 'rb').read(), channels=3)
#         img1 = tf.expand_dims(img1, 0)  
#         img1= transform_images(img1, size)
#         im=tf.concat((im,img1),axis=0)

#     print(tf.shape(im))


#     t1 = time.time()
#     Tboxes, Tscores, Tclasses, Tnums = yolo(im)
#     t2 = time.time()
#     logging.info('time: {}'.format(t2 - t1))
# #    boxes, scores, classes, nums=a
# #    print(tf.shape(Tscores[0:1,:]))
# #    print(tf.shape(Tclasses))
# #    print(tf.shape(Tboxes))
# #    print(tf.shape(Tnums[0:1]))



#     for pic in range(tf.shape(Tnums)):
#         scores=Tscores[0+pic:1+pic,:]
#         classes=Tclasses[0+pic:1+pic,:]
#         boxes=Tboxes[0+pic:1+pic,:,:]
#         nums=Tnums[0+pic:1+pic]


#         logging.info('detections:')
#         print(nums[0])
#         for i in range(nums[0]):
#             logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
#                                            scores[0][i].numpy(),
#                                            boxes[0][i].numpy()))


#         img = cv2.imread(images_url[pic])
#         img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
#         cv2.imwrite(str(pic)+'.jpg', img)
#         logging.info('output saved to: {}'.format('output'+str(pic)))

# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass