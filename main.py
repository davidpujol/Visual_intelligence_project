from segmentation.src.utils import create_model, process_image, bbox_per_person
from human_body_generation.utils import compute_new_image

# This file attempts to create the final pipeline for the project "Object substitution in RGB images"


# Create the segmentation model
seg_model = create_model()

# Define the input path
img_path = ''
threshold = 0.2

# Segment the image
orig_image, masks, boxes, labels = process_image(image_path=img_path, threshold=threshold, model= seg_model)

# Produce the new image in-painted by calling module 2


# Created the zoomed image for each of the human bodies in the image
person_images = bbox_per_person(orig_image, boxes, labels)

# Create the new image for each of the detected human bodies
new_images = []
for person_img in person_images:
    output_img = compute_new_image(person_img)
    new_images.append(output_img)

# Fuse the new images with the in-painted image
