import os
from .options.test_options import TestOptions
from .data_processing.data_loader import CreateDataLoader
from .models.models import create_model
from .tool.compute_coordinates_inference import compute_pose_estimation
#from util.visualizer import Visualizer
from skimage.io import imread
import matplotlib.pyplot as plt
import torch
from .util import util
from .data_processing.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Prepare the options for inference
def set_options_inference():
    opt = TestOptions(norm='batch', how_many=20, BP_input_nc=18, dataroot='./human_body_generation/market_data/',
                      name='market_PATN', nThreads=1, model='PATN', phase='test', dataset_mode='keypoint', batchSize=1,
                      serial_batches=False, no_flip=True, checkpoints_dir='./human_body_generation/checkpoints', which_model_netG='PATN',
                      pairLst='./human_body_generation/market_data/market-pairs-test.csv', results_dir='./human_body_generation/results', resize_or_crop='no', which_epoch='latest', display_id=0)
    return opt

# This function computes the pose estimation of a given image, and returns its corresponding pose.
def compute_pose(oriImg):
    # Prepare the training options
    opt = set_options_inference()
    transform = get_transform(opt)

    # Transform to B,G,R order
    oriImg2 = oriImg[:,:, ::-1]

    # Compute the pose estimation
    pose_img = compute_pose_estimation(oriImg2)

    # Reshape the original image into 1 x 3 x 128 x 64 (from 1 x 128 x 64 x 3)
    #P1 = Image.open(input_path).convert('RGB')

    # oriImg should be a PIL Image
    P1 = transform(oriImg.copy())
    P1 = torch.unsqueeze(P1, 0)

    return P1, pose_img


# This function create the dataset used to find the random poses. It also intializes the generative model
def create_generative_model():
    opt = set_options_inference()

    # Create the dataset from which we extract random poses
    data_loader = CreateDataLoader(opt)

    # dataset = data_loader.load_data()
    random_pose_dataset = data_loader.load_dataset()

    # Load the model
    gen_model = create_model(opt)
    gen_model = gen_model.eval()

    return random_pose_dataset, gen_model

# Computes the random pose by sampling u.a.r one of the data points of a given dataset.
# Then, it returns the pre-computed pose of the chosen image.
def compute_random_pose(dataset):
    rand_idx = np.random.randint(len(dataset))
    BP2 =  torch.unsqueeze(dataset[rand_idx]['BP1'], 0)

    #aux = util.draw_pose_from_map(BP2)[0]

    return BP2


# Provided the input image 1, the pose of image 1, and the pose of the new image (target pose), it returns the final image.
# Hence, given this information, it applies the pre-trained generative model
def apply_generative_model(model, img1, pose1, pose2):
    # Create the output_path (CHANGE THIS ONE)
    #output_path = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    # Predict the new image
    final_image = model.predict(img1, pose1, pose2)

    # Transform it to an image
    final_image = util.tensor2im(final_image.data)

    return final_image

# This is the main function, which puts of all of the previous functions together.
# This function takes as input an original image. Then, it computes the human pose of the input image
# and the target pose, by sampling from the train dataset for pre-computed poses.
# Finally, it returns the new, generated image.
def compute_new_image(oriImg):
    # Compute the pose of the original image
    P1, B1 = compute_pose(oriImg)

    # Initialize the model
    random_pose_dataset, gen_model = create_generative_model()

    # Compute the new target pose
    B2 = compute_random_pose(random_pose_dataset)
    B2 = F.interpolate(B2, size=(B1.shape[2], B1.shape[3]))
    # B2 = B2.resize_(1, 18, B1.shape[2], int(B1.shape[2]/B2.shape[2]*B2.shape[3]))

    #aux = util.draw_pose_from_map(B2)[0]
    #plt.imshow(aux)
    #plt.show()

    # Compute the final image
    P2 = apply_generative_model(gen_model, P1, B1, B2)

    return P2