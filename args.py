# Arguments used for networks
class Args_voc(object):
    def __init__(self):
        self.backbone = 'resnet'
        self.out_stride = 16
        self.dataset = 'pascal'
        self.use_sbd = False
        self.workers = 4
        self.base_size = (513, 513)
        self.crop_size = (513, 513)
        self.scale_ratio = (0.5, 2.0)  # random scale from 0.5 to 2.0
        self.sync_bn = None
        self.freeze_bn = False
        self.loss_type = 'ce'
        self.epochs = None
        self.start_epoch = 0
        self.batch_size = None
        self.test_batch_size = None
        self.use_balanced_weights = False
        self.lr = None
        self.lr_scheduler = 'poly'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.nesterov = False
        self.no_cuda = False
        self.gpu_ids = '0'
        self.seed = 1
        self.resume = None
        self.checkname = None
        self.ft = False
        self.eval_interval = 1
        self.no_val = False


class Args_occ5000(object):
    def __init__(self):
        self.backbone = 'resnet'
        self.out_stride = 16
        self.dataset = 'occ5000'
        self.use_sbd = False
        self.workers = 8
        self.base_size = (689, 161)#(1361, 305)  # scale on base_size from 0.5 to 2.0, should set to be same as image size
        self.crop_size = (689, 161)#(1361, 305)  # [h_crop, w_crop], crop_size = k * output_stride + 1, make crop_size as large as you can
        self.scale_ratio = (0.5, 2.0)  # random scale from 0.5 to 2.0
        self.sync_bn = None
        self.freeze_bn = False
        self.loss_type = 'ce'
        self.test_batch_size = None
        self.use_balanced_weights = False
        self.lr = None
        self.lr_scheduler = 'poly'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.nesterov = False
        self.no_cuda = False
        self.seed = 1
        self.resume = None  # '/home/kidd/kidd1/pytorch-deeplab-xception/run/occ5000/deeplab_kinematic/2020-01-19-01:06:06/35-0.497667546414435.pth.tar' # path to resume model file
        self.ft = False
        self.eval_interval = 1  # eval on eval set interval
        self.no_val = False
        self.network = 'hksl' #'hksl'
        # commonly used parameters
        # if self.network == 'deeplab':
        #     self.use_kinematic = False
        # elif self.network == 'hksl':
        #     self.use_kinematic = True
        self.use_kinematic = False
        self.no_flip = True  # flip images
        self.epochs = 50
        self.checkname = 'deeplab v3+_half_size' #'deeplab_v3+_noflip_occ5000_half_size_kinematic_10epochs_96c'
        self.gpu_ids = '0'
        self.start_epoch = 0
        self.batch_size = 8
        self.dataset_path = '/home/kidd/kidd1/Occ5000_half_size'

class Args_cihp(object):
    def __init__(self):
        self.backbone = 'resnet'
        self.out_stride = 16
        self.dataset = 'cihp'
        self.use_sbd = False
        self.workers = 8
        # base_size in the form of (h, w)
        self.base_size = (512, 512)#(1361, 305)  # scale on base_size from 0.5 to 2.0, should set to be same as image size (h, w)
        self.crop_size = (512, 512)#(1361, 305)  # [h_crop, w_crop], crop_size = k * output_stride + 1, make crop_size as large as you can
        self.scale_ratio = (0.5, 1.5)  # random scale from 0.5 to 2.0
        self.sync_bn = None
        self.freeze_bn = False
        self.loss_type = 'ce'
        self.test_batch_size = None
        self.use_balanced_weights = False
        self.lr = None
        self.lr_scheduler = 'poly'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.nesterov = False
        self.no_cuda = False
        self.seed = 1
        self.resume = None  # '/home/kidd/kidd1/pytorch-deeplab-xception/run/occ5000/deeplab_kinematic/2020-01-19-01:06:06/35-0.497667546414435.pth.tar' # path to resume model file
        self.ft = False
        self.eval_interval = 1  # eval on eval set interval
        self.no_val = False

        if self.dataset == 'occ5000':
            self.dataset_path = '/home/kidd/kidd1/Occ5000_half_size'
        self.use_kinematic = False
        self.no_flip = True  # flip images
        self.epochs = 50
        self.checkname = 'deeplabv3+' #'deeplab_v3+_noflip_occ5000_half_size_kinematic_10epochs_96c'
        self.gpu_ids = '0'
        self.start_epoch = 0
        self.batch_size = 8
        self.rotate_degree = 30

