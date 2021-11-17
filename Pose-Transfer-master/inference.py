import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from tool.compute_coordinates_inference import compute_pose_estimation
from util.visualizer import Visualizer
from skimage.io import imread
import matplotlib.pyplot as plt
import torch


# Prepare the options
def set_options_inference():
    opt = TestOptions(norm='batch', how_many=20, BP_input_nc=18, dataroot='./market_data/',
                      name='market_PATN', nThreads=1, model='PATN', phase='test', dataset_mode='keypoint', batchSize=1,
                      serial_batches=True, no_flip=True, checkpoints_dir='./checkpoints', which_model_netG='PATN',
                      pairLst='./market_data/market-pairs-test.csv', results_dir='./results', resize_or_crop='no', which_epoch='latest', display_id=0)
    return opt


# Provide the input image 1, the pose of image 1, and the pose of the new image
# Format: P1 image has shape torch.Size([1,3,128,64]) IMPORTANT!!
# BP1 has shape [1, 18,128, 64]. So it has 18 channels (Check how to generate this)
def process_image(opt, image1, pose1, pose2):
    # Load the model
    model = create_model(opt)

    # Create the visualizer
    visualizer = Visualizer(opt)

    # Create the output_path (CHANGE THIS ONE)
    output_path = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    # Predict the new image
    final_image = model.predict(image1, pose1, pose2)

    i = 0
    visualizer.save_images_custom(output_path, i, final_image)


def load_img(input_path):
    # Load the original image
    oriImg = imread(input_path)[:, :, ::-1]  # B,G,R order
    # plt.imshow(oriImg)
    # plt.show()
    # Create its pose estimation
    img_name = input_path.split('/')[-1]
    pose_img = compute_pose_estimation(oriImg, img_name)

    # What is the final representation of the pose?? Just the coordinates??
    # For now the representation is 128x64x18 (considering that the original image was 128x64)
    # Process the pose img
    pose_img = torch.from_numpy(pose_img).float()  # h, w, c
    print(pose_img.shape)
    pose_img = pose_img.transpose(2, 0)  # c,w,h
    pose_img = pose_img.transpose(2, 1)  # c,h,w

    # How do we generate a random pose??

    return img_name, pose_img, pose_img


# RUN: python inference.py
if __name__ == '__main__':
    # Prepare the training options
    opt = set_options_inference()

    # read the corresponding images
    img_path = os.path.abspath('./test_inference_img/0002_c1s1_000776_01.jpg')
    output_path = os.path.abspath('./')
    img1, pose1, pose2 = load_img(img_path)

    # process the images
    process_image(opt=opt, image1=img1, pose1=pose1, pose2=pose2)
