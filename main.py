from segmentation.src.utils import create_model, process_image, bbox_per_person, seg_person, seg_background
import human_body_generation.auxiliary_functions as human_body_generation
from DeepFillv2  import utils as df_utils
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import cv2
from PIL import Image

sys.path.append('./human_body_generation')

# This file attempts to create the final pipeline for the project "Object substitution in RGB images"


# Create the segmentation model
seg_model = create_model()

# Define the input path
#img_path = os.path.abspath('./segmentation/input/image1.jpg')
img_path = os.path.abspath('./human_body_generation/test_inference_img/test_image_3.jpg')
threshold = 0.9#0.965

# Segment the image
orig_image, masks, boxes, labels = process_image(image_path=img_path, threshold=0.5, model= seg_model)

'''
y_min, x_min = boxes[0][0]
y_max, x_max = boxes[0][1]
height_box = x_max - x_min
width_box = y_max - y_min

# Rescale the box (change this)
y_min -= (y - orig_length)
y_max -= (y - orig_length)
x_min -= (x - orig_width)
x_max -= (x - orig_width)

indices_of_mask = [(i+x_min, j+y_min, c) for i in range(height_box+1) for j in range(width_box) for c in range(channels) if seg_mask[i+x_min][j+y_min]]
exp_mask_bw =  np.zeros((height, width))
# Create the expanded mask
for i,j,c in indices_of_mask:
    exp_mask_bw[i][j] = True'''

tot_seg_mask = df_utils.concat_masks(masks)
print(tot_seg_mask, flush=True)
# Created the zoomed image for each of the human bodies in the image
person_images, person_boxes = bbox_per_person(orig_image, boxes, labels)

# Create the new image for each of the detected human bodies
new_images = []

# This is the final image with all the people masked out (on which we will paste the new images)
print("ORIGINAL IMAGE:", flush=True)
plt.imshow(orig_image)
plt.show()

output_image = seg_person(orig_image, masks, labels)
print("INITIAL OUTPUT IMAGE:", flush=True)
plt.imshow(output_image)
plt.show()

# Produce the new image in-painted by calling module 2
'''
print("impaint started", flush=True)
bg = df_utils.impainting(output_image, masks, use_gpu=True)
for m in masks:
    plt.imshow(m)
    plt.show()

print("showing image", flush=True)
plt.imshow(bg)
plt.show()
print("impaint done", flush=True)
'''

# Discard the first one since it contains the full image
#For visualisation purpose
i = 0
for person_img, person_box in list(zip(person_images, person_boxes)):
    i += 1
    print("PERSON IMAGE:", flush=True)
    plt.imshow(person_img.data)
    plt.show()
    person_img = np.asarray(person_img.data)

    # Problem: The new pose may not be of the size as the new image. Thus, we need to reshape it!!
    gen_img = human_body_generation.compute_new_image(person_img)
    print("GENERATED IMAGE", flush=True)
    plt.imshow(gen_img)

    save_dir = './human_body_generation/test_generated_image/image1' + '_generated'+'.jpg'

    # CHECK THIS! RIGHT NOW THE SEGMENTATION IS NOT DONE VERY WELL SINCE THE IMAGE IS TOO SMALL??
    # plt.savefig(save_dir)
    # resized = cv2.resize(gen_img, (int(gen_img.shape[1]*1.5), int(gen_img.shape[0]*1.5)), interpolation = cv2.INTER_AREA)

    img_to_save = Image.fromarray(gen_img)
    img_to_save.save(save_dir)
    plt.show()

    print("SHAPE OF THE IMAGE BEFORE PROCESSING", flush=True)
    # print(resized.shape)
    # orig_image_shape = resized.shape
    gen_img_shape = gen_img.shape
    generated_image, masks, gen_boxes, labels = process_image(image_path=save_dir, threshold=threshold, model= seg_model, image_type='gen')
    print("SHAPE OF THE IMAGE AFTER PROCESSING", flush=True)
    # Problem: Sometimes the number of masks is empty!!

    person_image_masked, seg_mask = seg_background(generated_image, masks, labels)
    x = int(np.array(person_image_masked).shape[0]/2)
    y = int(np.array(person_image_masked).shape[1]/2)
    orig_width = int(gen_img_shape[0]/2)
    orig_length = int(gen_img_shape[1]/2)
    #print(x, y, orig_width, orig_length)
    person_image_masked = person_image_masked[x-orig_width:x+orig_width, y-orig_length:y+orig_length,:]

    print(np.array(person_image_masked).shape, flush=True)
    print(gen_img_shape, flush=True)
    print("SHAPE OF MASKS: ", masks.shape, flush=True)
    #masks = masks[:, x-orig_width:x+orig_width, y-orig_length:y+orig_length]
    print(seg_mask.shape)
    seg_mask = seg_mask[x-orig_width:x+orig_width, y-orig_length:y+orig_length]
    #print(seg_mask.shape)
    #seg_mask =
    print("SHAPE OF CROPPED MASKS: ", seg_mask.shape, flush=True)

    print("GENERATED IMAGE SEGMENTED", flush=True)
    plt.imshow(person_image_masked)
    plt.show()

    print("GENERATED MASK", flush=True)
    print(seg_mask.shape, flush=True)
    plt.imshow(seg_mask)
    plt.show()

    # FUSION MODULE
    # Dimensions of the output_image
    height, width, channels = output_image.shape

    # Expand the person_image_masked to the same shape of the orig_image
    # Use its bounding box
    exp_mask = np.zeros((height, width, channels))

    y_min, x_min = gen_boxes[0][0]
    y_max, x_max = gen_boxes[0][1]
    height_box = x_max - x_min
    width_box = y_max - y_min

    # Rescale the box (change this)
    y_min -= (y - orig_length)
    y_max -= (y - orig_length)
    x_min -= (x - orig_width)
    x_max -= (x - orig_width)

    indices_of_mask = [(i+x_min, j+y_min, c) for i in range(height_box+1) for j in range(width_box) for c in range(channels) if seg_mask[i+x_min][j+y_min]]
    exp_mask_bw =  np.zeros((height, width))
    # Create the expanded mask
    for i,j,c in indices_of_mask:
        exp_mask[i][j][c] = 1
        exp_mask_bw[i][j] = True

    exp_mask_bw = exp_mask_bw.astype(bool)
    print("EXPANDED MASK", flush=True)
    plt.imshow(exp_mask)
    plt.show()

    # TODO: Create mask diff for impainting
    plt.imshow(exp_mask_bw)
    plt.show()
    print(tot_seg_mask.shape == exp_mask_bw.shape)
    diff_mask = tot_seg_mask
    dim = tot_seg_mask.shape
    print(exp_mask_bw, flush=True)
    for i in range(dim[0]):
        for j in range(dim[1]):
            if diff_mask[i][j]:
                if exp_mask_bw[i][j]:
                    print((diff_mask[i][j], exp_mask_bw[i][j]), flush=True)
                    diff_mask[i][j] = False
                else:
                    continue
            else:
                diff_mask[i][j] = False

    plt.imshow(diff_mask)
    plt.show()

    # Fuse both images
    # Apply the operation pixel wise: If mask[c][i][j] == 1, then use person_image_masked[c][i][j]. Else, use output_img[c][i][j]
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                if exp_mask[i][j][c]:
                    output_image[i][j][c] = gen_img[i][j][c]

    print("FUSED IMAGE", flush=True)
    plt.imshow(output_image)
    plt.show()

    # Produce the new image in-painted by calling module 2
    finished = df_utils.impainting(output_image, [diff_mask], use_gpu=True)
    plt.imshow(finished)
    plt.show()
    
    break




# Do the final impainting
