from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


# 1: Set the generator for the predictions.
img_height = 480 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 6 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True


# 1: Build the Keras model.

# K.clear_session() # Clear previous models from memory.

# model = ssd_300(image_size=(img_height, img_width, img_channels),
#                 n_classes=n_classes,
#                 mode='training',
#                 l2_regularization=0.0005,
#                 scales=scales,
#                 aspect_ratios_per_layer=aspect_ratios,
#                 two_boxes_for_ar1=two_boxes_for_ar1,
#                 steps=steps,
#                 offsets=offsets,
#                 clip_boxes=clip_boxes,
#                 variances=variances,
#                 normalize_coords=normalize_coords,
#                 subtract_mean=mean_color,
#                 swap_channels=swap_channels)

# # 2: Load some weights into the model.

# # TODO: Set the path to the weights you want to load.
# weights_path = 'ssd300_epoch2.h5'

# model.load_weights(weights_path, by_name=True)

# # 3: Instantiate an optimizer and the SSD loss function and compile the model.
# #    If you want to follow the original Caffe implementation, use the preset SGD
# #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# # sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)



# ============================================================================================

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'ssd300_epoch2.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                              'L2Normalization': L2Normalization,
                                              'DecodeDetections': DecodeDetections,
                                              'compute_loss': ssd_loss.compute_loss})

# ============================================================================================



# val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='ssd_120cat_val.h5')
# convert_to_3_channels = ConvertTo3Channels()
# resize = Resize(height=img_height, width=img_width)


# ========================================================================================================
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = 'detect/001624.jpeg'
print("img path:", img_path)


orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)

confidence_threshold = 0

y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
          'e2', 'm2', 'e1', 'm1',
          'e0', 'm0']

plt.figure(figsize=(20,12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / img_width
    ymin = box[3] * orig_images[0].shape[0] / img_height
    xmax = box[4] * orig_images[0].shape[1] / img_width
    ymax = box[5] * orig_images[0].shape[0] / img_height
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    print(label)
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})





# ============================================================================================================

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
# predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
#                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
#                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
#                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
#                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
#                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

# ssd_input_encoder = SSDInputEncoder(img_height=img_height,
#                                     img_width=img_width,
#                                     n_classes=n_classes,
#                                     predictor_sizes=predictor_sizes,
#                                     scales=scales,
#                                     aspect_ratios_per_layer=aspect_ratios,
#                                     two_boxes_for_ar1=two_boxes_for_ar1,
#                                     steps=steps,
#                                     offsets=offsets,
#                                     clip_boxes=clip_boxes,
#                                     variances=variances,
#                                     matching_type='multi',
#                                     pos_iou_threshold=0.5,
#                                     neg_iou_limit=0.5,
#                                     normalize_coords=normalize_coords)


# predict_generator = val_dataset.generate(batch_size=16,
#                                          shuffle=True,
#                                          transformations=[convert_to_3_channels,
#                                                           resize],
#                                          label_encoder=ssd_input_encoder,
#                                          returns={'processed_images',
#                                                   'filenames',
#                                                   'inverse_transform',
#                                                   'original_images',
#                                                   'original_labels'},
#                                          keep_images_without_gt=False)

# # 2: Generate samples.

# batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

# i = 0 # Which batch item to look at
# print(len(batch_filenames))
# print("Image:", batch_filenames[i])
# print()
# print("Ground truth boxes:\n")
# print(np.array(batch_original_labels[i]))

# # 3: Make predictions.

# y_pred = model.predict(batch_images)

# # 4: Decode the raw predictions in `y_pred`.

# y_pred_decoded = decode_detections(y_pred,
#                                   confidence_thresh=0.5,
#                                   iou_threshold=0.4,
#                                   top_k=200,
#                                   normalize_coords=normalize_coords,
#                                   img_height=img_height,
#                                   img_width=img_width)

# # 5: Convert the predictions for the original image.

# y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('   class   conf xmin   ymin   xmax   ymax')
# print(y_pred_decoded_inv[i])

# # 5: Draw the predicted boxes onto the image

# # Set the colors for the bounding boxes
# colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
# classes = ['background',
#           'e2', 'm2', 'e1', 'm1',
#           'e0', 'm0']

# plt.figure(figsize=(20,12))
# plt.imshow(batch_original_images[i])

# current_axis = plt.gca()

# for box in batch_original_labels[i]:
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

# for box in y_pred_decoded_inv[i]:
#     xmin = box[2]
#     ymin = box[3]
#     xmax = box[4]
#     ymax = box[5]
#     color = colors[int(box[0])]
#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})