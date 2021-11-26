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
     checkpoints_dir = attr.ib(default='./checkpoints')
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
     pairLst = attr.ib(default='./keypoint_data/market-pairs-train.csv')

     with_D_PP = attr.ib(default=1)
     with_D_PB = attr.ib(default=1)

     use_flip = attr.ib(default=0)

     # down-sampling times
     G_n_downsampling = attr.ib(default=2)
     D_n_downsampling = attr.ib(default=2)

     # Specific training options
     ntest = attr.ib(default=float("inf"))
     results_dir = attr.ib(default='./results/')
     aspect_ratio = attr.ib(default=1.0)
     phase = attr.ib(default='test')
     which_epoch = attr.ib(default='latest')
     how_many = attr.ib(default=200)
     isTrain = attr.ib(default=False)



# class TestOptions(BaseOptions):
#       def initialize(self):
#           BaseOptions.initialize(self)
#           self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
#           self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
#           self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
#           self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
#           self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
#           self.parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
#
#           self.isTrain = False
#           self.use_gpus =False