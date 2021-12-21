import os
from options.test_options import TestOptions
from data_processing.data_loader import CreateDataLoader
from models.models import create_model
from tool.compute_coordinates_inference import compute_pose_estimation
from util.visualizer import Visualizer
from skimage.io import imread
import matplotlib.pyplot as plt
import torch
from util import util
from data_processing.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np

# Prepare the options for inference
def set_options_inference():
    opt = TestOptions(norm='batch', how_many=20, BP_input_nc=18, dataroot='./market_data/',
                      name='market_PATN', nThreads=1, model='PATN', phase='test', dataset_mode='keypoint', batchSize=1,
                      serial_batches=False, no_flip=True, checkpoints_dir='./checkpoints', which_model_netG='PATN',
                      pairLst='./market_data/market-pairs-test.csv', results_dir='./results', resize_or_crop='no', which_epoch='latest', display_id=0)
    return opt

# Load the original image given its path. Then it applies a transformation to convert it into a tensor.
def load_img(input_path, transform):
    # Load the original image
    oriImg = imread(input_path)[:,:, ::-1]# B,G,R order

    # Computes the pose estimation
    pose_img = compute_pose_estimation(oriImg)

    # Reshape the original image into 1 x 3 x 128 x 64 (from 1 x 128 x 64 x 3)
    P1 = Image.open(input_path).convert('RGB')
    P1 = transform(P1.copy())
    P1 = torch.unsqueeze(P1, 0)

    return P1, pose_img

# Computes the random pose by sampling u.a.r one of the data points of a given dataset.
# Then, it returns the pre-computed pose of the chosen image.
def compute_random_pose(dataset):
    rand_idx = np.random.randint(len(dataset))
    BP2 =  torch.unsqueeze(dataset[rand_idx]['BP1'], 0)

    return BP2


# Provided the input image 1, the pose of image 1, and the pose of the new image (target pose), it returns the final image.
# Hence, given this information, it applies the pre-trained generative model
def process_image(opt, image1, pose1, pose2):
    # Load the model
    model = create_model(opt)
    model = model.eval()

    # Predict the new image
    final_image = model.predict(image1, pose1, pose2)

    # Transform it to an image
    final_image = util.tensor2im(final_image.data)
    #plt.imshow(final_image.data)
    #plt.show()

    return final_image


# RUN: python inference.py
if __name__ == '__main__':
    # Prepare the training options
    opt = set_options_inference()
    transform = get_transform(opt)

    # Create the dataset from which we extract random poses
    data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    dataset = data_loader.load_dataset()

    # read the corresponding images
    img_path = os.path.abspath('./test_inference_img/image1.jpg')
    output_path = os.path.abspath('./')
    img1, pose1 = load_img(img_path, transform)

    pose2 = compute_random_pose(dataset)

    # process the images
    final_image = process_image(opt=opt, image1=img1, pose1=pose1, pose2=pose2)

    plt.imshow(final_image.data)
    plt.show()