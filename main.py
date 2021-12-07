from segmentation.src.utils import create_model, process_image, bbox_per_person
from human_body_generation.utils import compute_new_image
import matplotlib.pyplot as plt
import os

import numpy as np

from segmentation.src.coco_names import COCO_INSTANCE_CATEGORY_NAMES
from segmentation.src.utils import create_model, process_image, bbox_per_person
#from human_body_generation.utils import compute_new_image
from DeepFillv2.impaint import impaint
from DeepFillv2.utils import concat_masks
import cv2
from PIL import Image
# This file attempts to create the final pipeline for the project "Object substitution in RGB images"


# Create the segmentation model
seg_model = create_model()

# Define the input path
img_path = 'DeepFillv2/test_data/1.png'

#img_path = os.path.abspath('./human_body_generation/test_inference_img/0002_c1s1_000551_01.jpg')
threshold = 0.965

# Segment the image
orig_image, masks, boxes, labels = process_image(image_path=img_path, threshold=threshold, model= seg_model)

# Produce the new image in-painted by calling module 2
full_mask = concat_masks(masks)
cv2.imwrite('masks/full.png', full_mask)

bg = impaint(orig_image, full_mask)

'''
# Created the zoomed image for each of the human bodies in the image
person_images = bbox_per_person(orig_image, boxes, labels)

# Create the new image for each of the detected human bodies
new_images = []

# Discard the first one since it contains the full image
for person_img in person_images[1:]:
    plt.imshow(person_img.data)
    plt.show()
    person_img = np.asarray(person_img.data)
    output_img = compute_new_image(person_img)
<<<<<<< HEAD
    new_images.append(output_img)
'''
    #plt.imshow(output_img)
    #plt.show()
    #new_images.append(output_img)

# Fuse the new images with the in-painted image
