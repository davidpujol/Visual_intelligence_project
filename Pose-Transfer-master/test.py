import time
import os
from options.test_options import TestOptions
from data_processing.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time

opt = TestOptions(norm='batch', how_many=20, BP_input_nc=18, dataroot='./market_data/',
                   name='market_PATN', nThreads=1, model='PATN', phase='test', dataset_mode='keypoint', batchSize=1,
                   serial_batches=True, no_flip=True, checkpoints_dir='./checkpoints', which_model_netG='PATN',
                   pairLst='./market_data/market-pairs-test.csv', results_dir='./results', resize_or_crop='no', which_epoch='latest', display_id=0)



data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = model.eval()

#opt.how_many = 999999
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    print(endTime-startTime)
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    img_path = [img_path]

    visualizer.save_images(webpage, visuals, img_path)

webpage.save()




