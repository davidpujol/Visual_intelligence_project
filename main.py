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
img_path = os.path.abspath('./human_body_generation/test_inference_img/image1.jpg')
threshold = 0.9#0.965

# Segment the image
orig_image, masks, boxes, labels = process_image(image_path=img_path, threshold=threshold, model= seg_model)

# Produce the new image in-painted by calling module 2
#bg = df_utils.impainting(orig_image, masks, -1, use_gpu=True)

# Created the zoomed image for each of the human bodies in the image
person_images, person_boxes = bbox_per_person(orig_image, boxes, labels)

# Create the new image for each of the detected human bodies
new_images = []

# This is the final image with all the people masked out (on which we will paste the new images)
print("ORIGINAL IMAGE:")
plt.imshow(orig_image)
plt.show()

output_image = seg_person(orig_image, masks, labels)
print("INITIAL OUTPUT IMAGE:")
plt.imshow(output_image)
plt.show()
    
# Discard the first one since it contains the full image
#For visualisation purpose
i = 0
for person_img, person_box in list(zip(person_images, person_boxes))[0:]:
    i += 1
    print("PERSON IMAGE:")
    plt.imshow(person_img.data)
    plt.show()
    person_img = np.asarray(person_img.data)

    # Problem: The new pose may not be of the size as the new image. Thus, we need to reshape it!!
    gen_img = human_body_generation.compute_new_image(person_img)
    print("GENERATED IMAGE")
    plt.imshow(gen_img)

    save_dir = './human_body_generation/test_generated_image/image1' + '_generated'+'.jpg'

    # CHECK THIS! RIGHT NOW THE SEGMENTATION IS NOT DONE VERY WELL SINCE THE IMAGE IS TOO SMALL??
    # plt.savefig(save_dir)
    # resized = cv2.resize(gen_img, (int(gen_img.shape[1]*1.5), int(gen_img.shape[0]*1.5)), interpolation = cv2.INTER_AREA)

    img_to_save = Image.fromarray(gen_img)
    img_to_save.save(save_dir)
    plt.show()

    print("SHAPE OF THE IMAGE BEFORE PROCESSING")
    # print(resized.shape)
    # orig_image_shape = resized.shape
    gen_img_shape = gen_img.shape
    generated_image, masks, gen_boxes, labels = process_image(image_path=save_dir, threshold=threshold, model= seg_model, image_type='gen')
    # TODO: remove comment when function is back
    print("SHAPE OF THE IMAGE AFTER PROCESSING")
    # Problem: Sometimes the number of masks is empty!!

    person_image_masked, seg_mask = seg_background(generated_image, masks, labels)
    x = int(np.array(person_image_masked).shape[0]/2)
    y = int(np.array(person_image_masked).shape[1]/2)
    orig_width = int(gen_img_shape[0]/2)
    orig_length = int(gen_img_shape[1]/2)
    print(x, y, orig_width, orig_length)
    person_image_masked = person_image_masked[x-orig_width:x+orig_width, y-orig_length:y+orig_length,:]
    print(np.array(person_image_masked).shape)
    print(gen_img_shape)

    print("GENERATED IMAGE SEGMENTED")
    
    plt.imshow(person_image_masked)
    plt.show()
    quit()

    print("GENERATED MASK")
    print(seg_mask.shape)
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

    indices_of_mask = [(i+x_min, j+y_min, c) for i in range(height_box+1) for j in range(width_box) for c in range(channels) if seg_mask[i+x_min][j+y_min]]
    # Create the expanded mask
    for i,j,c in indices_of_mask:
        exp_mask[i][j][c] = 1

    print("EXPANDED MASK")
    plt.imshow(exp_mask)
    plt.show()

    # Fuse both images
    # Apply the operation pixel wise: If mask[c][i][j] == 1, then use person_image_masked[c][i][j]. Else, use output_img[c][i][j]
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                if exp_mask[i][j][c]:
                    output_image[i][j][c] = gen_img[i][j][c]

    print("FUSED IMAGE")
    plt.imshow(output_image)
    plt.show()

    # Produce the new image in-painted by calling module 2
    #bg = df_utils.impainting(orig_image, masks, i, use_gpu=False)






# Do the final impainting