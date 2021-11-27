#from .base_options import BaseOptions
import attr

@attr.s
class TestOptions:
     dataroot = attr.ib(None)
     batchSize = attr.ib(default=1)
     loadSize = attr.ib(default=286)
     fineSize = attr.ib(default=256)
     input_nc = attr.ib(default=3)
     output_nc = attr.ib(default=3)
     ngf = attr.ib(default=64)
     ndf = attr.ib(default=64)
     which_model_netD = attr.ib(default='resnet')
     which_model_netG = attr.ib(default='PATN')
     n_layers_D = attr.ib(default=3)
     # gpu_ids = attr.ib(default=[])
     use_gpus = attr.ib(default=False)
     name = attr.ib(default='experiment_name')
     dataset_mode = attr.ib(default='unaligned')
     model = attr.ib(default='cycle_gan')
     which_direction = attr.ib(default='AtoB')
     nThreads = attr.ib(default=2)
     checkpoints_dir = attr.ib(default='./human_body_generation/checkpoints')
     norm = attr.ib(default='instance')
     serial_batches = attr.ib(default=True)
     display_winsize = attr.ib(default=256)
     display_id = attr.ib(default=1)
     display_port = attr.ib(default=8097)
     no_dropout = attr.ib(default=False)
     max_dataset_size = attr.ib(default=float("inf"))
     resize_or_crop = attr.ib(default='resize_and_crop')
     no_flip = attr.ib(default=True)
     init_type = attr.ib(default='normal')

     P_input_nc = attr.ib(default=3)
     BP_input_nc = attr.ib(default=1)
     padding_type = attr.ib(default='reflect')
     pairLst = attr.ib(default='./human_body_generation/keypoint_data/market-pairs-train.csv')

     with_D_PP = attr.ib(default=1)
     with_D_PB = attr.ib(default=1)

     use_flip = attr.ib(default=0)

     # down-sampling times
     G_n_downsampling = attr.ib(default=2)
     D_n_downsampling = attr.ib(default=2)

     # Specific training options
     ntest = attr.ib(default=float("inf"))
     results_dir = attr.ib(default='./human_body_generation/results/')
     aspect_ratio = attr.ib(default=1.0)
     phase = attr.ib(default='test')
     which_epoch = attr.ib(default='latest')
     how_many = attr.ib(default=200)
     isTrain = attr.ib(default=False)


