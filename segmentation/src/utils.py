from PIL.Image import new
import cv2
import numpy as np
import random
import torch
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels

def split_person(image, masks, labels):

    color = np.array([255,255,255])
    alpha = 1 
    beta = 1# transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        if labels[i] != 'person':
            continue
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)

        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

    return image

def bbox_per_person(image, bboxs, labels):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    person_bboxs = []
    person_images = []
    person_images.append(image)
    for i in range(len(bboxs)):
        if labels[i] == 'person':
            person_bboxs.append(bboxs[i])
    for box in person_bboxs:
        crop = image[box[0][1]:box[1][1],box[0][0]:box[1][0],:]
        person_images.append(crop)

    return person_images

def split_per_person(image, masks, labels):
    color = np.array([255,255,255])
    alpha = 1 
    beta = 1# transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    images = []
    image = np.array(image)
    person_masks = []
    images.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


    for i in range(len(masks)):
        if labels[i] == 'person':
            person_masks.append(masks[i])
    
    for i in range(len(person_masks)):
        new_mask = np.zeros_like(person_masks[i]).astype(np.uint8)
        for j in range(len(person_masks)):
            if j != i:
                new_mask = new_mask | person_masks[j]
        red_map = np.zeros_like(new_mask).astype(np.uint8)
        green_map = np.zeros_like(new_mask).astype(np.uint8)
        blue_map = np.zeros_like(new_mask).astype(np.uint8)
        red_map[new_mask == 1], green_map[new_mask == 1], blue_map[new_mask == 1]  = color
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)

        # convert from RGN to OpenCV BGR format
        image_per_person = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # apply mask on the image
        cv2.addWeighted(image_per_person, alpha, segmentation_map, beta, gamma, image_per_person)
        images.append(image_per_person)

    return images

# def construct_seg_map(person_masks, i):
#     seg_maps = []
#     for i in range(len(person_masks)):





def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        print(color)
        print(type(color))
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        print('segmentation map shape: ' + str(segmentation_map.shape))
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image