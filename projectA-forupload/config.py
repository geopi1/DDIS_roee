import os
from easydict import EasyDict as edict
from CX.enums import Distance
import tensorflow as tf
import re

celebA = False
single_image = False
zero_tensor = tf.constant(0.0, dtype=tf.float32)

# create a edict accessing them with dict.X
config = edict()


# ---------------------------------------------
#               update paths
config.base_dir = os.path.join('C:\\', 'Users', 'eyal', 'Desktop', 'projectA', 'vot_full')
config.vgg_model_path = os.path.join(config.base_dir, 'VGG_TRAIN', 'imagenet-vgg-verydeep-19.mat')
# ---------------------------------------------


config.W = edict()
# weights
config.W.CX = 1.0
config.W.CX_content = 1.0

# train parameters
config.TRAIN = edict()
config.TRAIN.is_train = False  # change to True if you want to train
config.TRAIN.sp = 256
config.TRAIN.aspect_ratio = 1  # 1
config.TRAIN.resize = [config.TRAIN.sp * config.TRAIN.aspect_ratio, config.TRAIN.sp]
config.TRAIN.crop_size = [config.TRAIN.sp * config.TRAIN.aspect_ratio, config.TRAIN.sp]
config.TRAIN.A_data_dir = 'Train'
config.TRAIN.out_dir = 'Result'
config.TRAIN.num_epochs = 1000
config.TRAIN.save_every_nth_epoch = 100
config.TRAIN.reduce_dim = 2  # use of smaller CRN model
config.TRAIN.every_nth_frame = 100  # train using all frames
config.TRAIN.epsilon = 1e-10

config.TRAIN.models = os.path.join(config.base_dir, config.TRAIN.out_dir, '0010')

config.VAL = edict()
config.VAL.A_data_dir = 'Validate'
config.VAL.every_nth_frame = 100

config.TEST = edict()
config.TEST.is_test = not config.TRAIN.is_train  # test only when not training
config.TEST.A_data_dir = "Test"
config.TEST.is_resize = True
config.TEST.is_fliplr = False
config.TEST.jump_size = 20
config.TEST.random_crop = False  # if False, take the top left corner

config.CX = edict()
config.CX.crop_quarters = False
config.CX.max_sampling_1d_size = 65
config.CX.feat_layers = {'conv1_1': 1.0,'conv2_1': 1.0, 'conv3_1': 1.0}
#config.CX.feat_layers = {'conv1_1': 1.0,'conv2_1': 1.0, 'conv3_1': 1.0, 'conv4_1': 1.0,'conv5_1': 1.0}
#config.CX.feat_layers = {'conv2_2': 1.0, 'conv3_3': 1.0}
config.CX.feat_content_layers = {'conv3_1': 1.0}  # for single image
config.CX.Dist = Distance.DotProduct
config.CX.nn_stretch_sigma = 0.5  # 0.1
config.CX.patch_size = 5
config.CX.patch_stride = 2


def last_two_nums(str):
    if str.endswith('vgg_input_im') or str is 'RGB':
        return 'rgb'
    all_nums = re.findall(r'\d+', str)
    return all_nums[-2] + all_nums[-1]


config.TEST.out_dir = config.TRAIN.out_dir
if not os.path.exists(config.TRAIN.out_dir):
    os.makedirs(config.TRAIN.out_dir)


