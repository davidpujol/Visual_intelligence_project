from segmentation.src.utils import create_model, process_image, bbox_per_person, seg_person, seg_background
import human_body_generation.auxiliary_functions as human_body_generation
from DeepFillv2 import utils as df_utils
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import cv2
from PIL import Image
sys.path.append('./human_body_generation')

# Produces the final cropped image of the person (after doing the necessary rescaling), as well as its segmentation mask
def produce_image_masked(generated_image, masks, labels, gen_img):

    gen_img_shape = gen_img.shape

    # Obtain the person (masking out the background)
    person_image_masked, seg_mask = seg_background(generated_image, masks, labels)

    # Rescales it to the correct shape
    x = int(np.array(person_image_masked).shape[0] / 2)
    y = int(np.array(person_image_masked).shape[1] / 2)
    orig_width = int(gen_img_shape[0] / 2)
    orig_length = int(gen_img_shape[1] / 2)

    # Crop the image and the mask correspondingly
    person_image_masked = person_image_masked[x - orig_width:x + orig_width, y - orig_length:y + orig_length, :]
    seg_mask = seg_mask[x - orig_width:x + orig_width, y - orig_length:y + orig_length]

    return person_image_masked, seg_mask, x, y, orig_length, orig_width

# Fuses both the generated human body and the current output image, which masked out the initial human bodies.
def fuse_images(output_image, gen_img, seg_mask, gen_boxes, x, y, orig_length, orig_width):
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

    indices_of_mask = [(i + x_min, j + y_min, c) for i in range(height_box + 1) for j in range(width_box) for c in
                       range(channels) if seg_mask[i + x_min][j + y_min]]
    exp_mask_bw = np.zeros((height, width))

    # Create the expanded mask
    for i, j, c in indices_of_mask:
        exp_mask[i][j][c] = 1
        exp_mask_bw[i][j] = True

    exp_mask_bw = exp_mask_bw.astype(bool)

    # Fuse both images
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                if exp_mask[i][j][c]:
                    output_image[i][j][c] = gen_img[i][j][c]

    #print("FUSED IMAGE", flush=True)
    #plt.imshow(output_image)
    #plt.show()

    return output_image, exp_mask_bw

def expand_mask(mask):
    height, width = mask.shape
    exp_mask = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            masked = False
            for pad in range(1, 3):
                if i - pad >= 0:
                    masked = masked | mask[i - pad][j]
                if i + pad < height:
                    masked = masked | mask[i + pad][j]
                if j - pad >= 0:
                    masked = masked | mask[i][j - pad]
                if j + pad < width:
                    masked = masked | mask[i][j + pad]
            exp_mask[i][j] = masked

    return exp_mask


# Computes the mask of the inpainting, this being, the mask that identifies the missing pixels in the image after doing the fusion
def compute_mask_inpainting(exp_mask_bw, tot_seg_mask):
    diff_mask = tot_seg_mask
    dim = tot_seg_mask.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if diff_mask[i][j]:
                if exp_mask_bw[i][j]:
                    #print((diff_mask[i][j], exp_mask_bw[i][j]), flush=True)
                    diff_mask[i][j] = False
                else:
                    continue
            else:
                diff_mask[i][j] = False

    plt.imshow(diff_mask)
    plt.show()

    return diff_mask

# This is the main functionality that implements the whole pipeline of our proposal for the human body substitution
def generate_image(img_path):
    # Create the segmentation model
    seg_model = create_model()

    # Define the threshold
    threshold = 0.9

    # Segment the image
    orig_image, masks, boxes, labels = process_image(image_path=img_path, threshold=0.5, model= seg_model)

    # Fuse all the masks from the segmentation
    tot_seg_mask = df_utils.concat_masks(masks)

    # Expand the mask to completely remove the person
    tot_seg_mask = expand_mask(tot_seg_mask)

    # Inpaint first to show bad results
    inpaint = df_utils.impainting(orig_image, [tot_seg_mask], use_gpu=True)
    cv2.imwrite("./results/first_inpaint.png", inpaint)
    plt.imshow(inpaint)
    plt.show()

    # Create the bounding boxes of each of the human bodies, and crop the corresponding image
    person_images, person_boxes = bbox_per_person(orig_image, boxes, labels)

    # Plot the original image
    print("ORIGINAL IMAGE:", flush=True)
    plt.imshow(orig_image)
    plt.show()

    # Define the image where all the human bodies are segmented out
    output_image = seg_person(orig_image, masks, labels)

    # Discard the first one since it contains the full image
    # Iterate over all the people in the image
    i = 0
    for person_img, person_box in list(zip(person_images, person_boxes)):
        i += 1

        # Transform the image into a numpy array
        person_img = np.asarray(person_img.data)

        # Calls the generative module, which creates a new generated image given the original cropped image of a human body
        gen_img = human_body_generation.compute_new_image(person_img)

        # Define the path where the generated image is saved
        save_dir = './human_body_generation/test_generated_image/image1' + '_generated'+'.jpg'

        # Save the generated image
        img_to_save = Image.fromarray(gen_img)
        img_to_save.save(save_dir)

        # Applies the segmentation module over the generated image, which allows us to
        # get rid of the generated background, and deal only with the new human poses
        generated_image, masks, gen_boxes, labels = process_image(image_path=save_dir, threshold=threshold, model= seg_model, image_type='gen')

        # Generate the cropped image for each of the people in the image, as well as its corresponding segmented mask
        person_image_masked, seg_mask, x, y, orig_length, orig_width = produce_image_masked(generated_image, masks, labels, gen_img)

        # This calls the fuse model to update the output image, by combining the current output image and the new generated human body
        output_image, exp_mask_bw = fuse_images(output_image, gen_img, seg_mask, gen_boxes, x, y, orig_length, orig_width)

        # Create mask diff for impainting
        diff_mask = compute_mask_inpainting(exp_mask_bw, tot_seg_mask)

        # Produce the new image in-painted by calling module 2
        finished = df_utils.impainting(output_image, [diff_mask], use_gpu=True)
        plt.imshow(finished)
        plt.show()

        break


if __name__ == '__main__':
    img_path = os.path.abspath('./DeepFillv2/test_data/1.png')
    generate_image(img_path)
