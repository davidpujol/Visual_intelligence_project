#from .base_options import BaseOptions
import attr

@attr.s
class TrainOptions:
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
    #gpu_ids = attr.ib(default=[])
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
    no_dropout = attr.ib(default=True)
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


    # Training options
    display_freq = attr.ib(default=100)
    display_single_pane_ncols = attr.ib(default=0)
    update_html_freq = attr.ib(default=1000)
    print_freq = attr.ib(default=100)
    save_latest_freq = attr.ib(default=5000)
    save_epoch_freq = attr.ib(default= 20)
    continue_train = attr.ib(default=False)
    epoch_count = attr.ib(default=1)
    phase = attr.ib(default='train')
    which_epoch = attr.ib(default='latest')
    niter = attr.ib(default=100)
    niter_decay = attr.ib(default=100)
    beta1 = attr.ib(default=0.5)
    lr = attr.ib(default=0.0002)
    no_lsgan = attr.ib(default=False)
    lambda_A = attr.ib(default=10.0)
    lambda_B = attr.ib(default=10.0)
    lambda_GAN = attr.ib(default=5.0)

    pool_size = attr.ib(default=50)
    no_html = attr.ib(default=True)
    lr_policy = attr.ib(default='lambda')
    lr_decay_iters = attr.ib(default=50)
    no_dropout_D = attr.ib(default=True)

    L1_type = attr.ib(default='origin')
    perceptual_layers = attr.ib(default=3)
    percep_is_l1 = attr.ib(default=1)
    DG_ratio = attr.ib(default=1)
    isTrain = attr.ib(default=True)



# class TrainOptions(BaseOptions):
#     def initialize(self):
#         BaseOptions.initialize(self)
#         self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
#         self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
#         self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
#         self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
#         self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
#         self.parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
#         self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
#         self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
#         self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
#         self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
#         self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
#         self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
#         self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
#         self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
#         self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
#         self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for L1 loss')
#         self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for perceptual L1 loss')
#         self.parser.add_argument('--lambda_GAN', type=float, default=5.0, help='weight of GAN loss')
#
#         self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
#         self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
#         self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
#         self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
#
#         self.parser.add_argument('--L1_type', type=str, default='origin', help='use which kind of L1 loss. (origin|l1_plus_perL1)')
#         self.parser.add_argument('--perceptual_layers', type=int, default=3, help='index of vgg layer for extracting perceptual features.')
#         self.parser.add_argument('--percep_is_l1', type=int, default=1, help='type of perceptual loss: l1 or l2')
#         self.parser.add_argument('--no_dropout_D', action='store_true', help='no dropout for the discriminator')
#         self.parser.add_argument('--DG_ratio', type=int, default=1, help='how many times for D training after training G once')
#
#
#
#
#         self.isTrain = True
