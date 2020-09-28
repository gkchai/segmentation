MODELS = ['unet', 'unet_dilated', 'fcn'] # list of available models
LOSSES = ['dice', 'jacc', 'sce']  # list of available loss metrics
OPTIMIZERS = ['adam', 'sgd', 'rmsprop', 'adagrad']  # list of available optimizers
AUGMENTATIONS = ['flip_h', 'flip_v', 'hue', 'brightness', 'saturation'] # list of possible augmentations
DIMS = [128, 256]
TRAIN_DATSETS = ['isic17', 'isic18', 'usr_mob', 'usr_d415', 'tones'] # list of train datasets
VAL_DATASETS = ['isic17', 'isic18', 'usr_mob', 'usr_d415', 'tones'] # list of val datasets
TEST_DATASETS = ['isic17', 'usr_mob', 'tones'] # list of test datasets
WEIGHTS_DATASETS = {'isic17':1.0, 'isic18':1.0, 'usr_mob':1.0, 'usr_d415':1.0, 'tones':0.5} # map for dataset->weights


class Config:
    def __init__(self):
        # default config
        self.num_layers = 4  # number of layers in encoding/decoding block
        self.init_channels = 16  # no. of channels in the first layer
        self.model = 'unet'  # name of the model
        self.batch_norm = True  # use batch norm in training
        self.gaussian_noise = False  # add gaussian noise in training
        self.l2_regularizer = True # use L2 regularizer for convolutional kernel
        self.dropout = False  # use dropout after pooling
        self.height = 128    # input image height
        self.width = 128    # input image width
        self.dim = 128  # input image dim (same as height and width)
        self.batch_size = 4  # number of images in batch
        self.lr_decay = 0.9  # learning rate decay
        self.seed = 31415   # fixed seed
        self.export_dir = 'export_dir'  # directory for exported model files
        self.ck_dir = 'ckdir'  # directory for model checkpoints
        self.out_dir = 'output'  # directory for evaluation output
        self.data_dir = 'dataset'  # directory containing datasets
        self.log_dir = 'logdir'  # directory for logging summary
        self.latest_ckpt_file = 'best_checkpoint'  # meta file pointing to last checkpoint to use
        self.moving_window = 1  # smoothing window size for calculating training scores
        self.patience = 15 # number of epochs with no improvement after which training will be stopped
        self.min_delta = 0.0001 # minimum change in the monitored quantity to qualify as an improvement
        self.model_basename = 'seg_model'
        self.best_model_basename = 'best_seg_model'
        self.weights_data = WEIGHTS_DATASETS

    @staticmethod
    def get_options():
        # list of options that will be parsed from commmand line
        options = \
        [
            # (arg, type, default value, help, choices)
            ('model_id', str, '','unique id of the model', None),
            ('exp', str, 'trial', 'experiment name', None),
            ('learning_rate', float, 1e-3, 'learning rate', None),
            ('dim', int, None, 'image dimension', DIMS),
            ('nb_epochs', int, 100, 'maximum number of training epochs', None),
            ('model', str, '', 'type of model', MODELS),
            ('ksize', int, 5, 'filter kernel size', None),
            ('train_data', list, ['isic17', 'isic18', 'usr_mob', 'usr_d415', 'tones'], 'train csv files',
             TRAIN_DATSETS),
            ('val_data', list, ['isic17', 'usr_mob'], 'val csv files', VAL_DATASETS),
            ('test_data', list, ['isic17', 'usr_mob'], 'test csv files', TEST_DATASETS),
            ('restart', bool, False, 'Delete checkpoint and restart training', None),
            ('max_to_keep', int, 1, 'Max # of checkpoints to keeps around', None),
            ('save_period', int, 10, 'Epoch period to save the trained model', None),
            ('phase', str, 'train', 'Phase: Can be train or test', ['train', 'test']),
            ('augment', list, ['flip_h', 'flip_v'], 'list of augmentations', AUGMENTATIONS),
            ('loss', str, 'dice', 'loss metric',  LOSSES),
            ('optimizer', str, 'adam', 'choice of optimizer', OPTIMIZERS)
        ]
        return options

    @staticmethod
    def get_dataset_keys():
        # list of dataset keys from config
        return ['train_data', 'val_data', 'test_data', 'weights_data']

    @staticmethod
    def get_params_keys():
        # list of hyperparameter keys from config
        return ['augment', 'batch_norm', 'batch_size', 'dim', 'height', 'init_channels', 'ksize',
                'l2_regularizer', 'dropout', 'learning_rate', 'loss', 'lr_decay', 'model', 'num_layers',
                'optimizer', 'moving_window', 'patience', 'min_delta', 'seed', 'width', 'exp']

    @staticmethod
    def get_metrics_keys():
        # list of metrics keys
        return ['best_epoch', 'n_train', 'n_val', 'train_data', 'val_data', 'val_scores',
                'train_scores','each_val_scores', 'each_train_scores']

    @staticmethod
    def get_common_keys():
        # list of common keys from config
        return ['model_id', 'curr_epoch', 'nb_epochs']

    @staticmethod
    def get_hash_keys():
        # list of keys for hashing the model config
        return Config.get_params_keys() + Config.get_dataset_keys()