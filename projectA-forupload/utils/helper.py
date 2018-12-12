# ---------------------------------------------------
#   code credits: https://github.com/CQFIO/PhotographicImageSynthesis
# ---------------------------------------------------
import numpy as np
import scipy
import scipy.misc
from config import *
import tensorflow as tf
import matplotlib.pyplot as plt


def read_image(file_name, resize=True, fliplr=False):
    image = np.float32(scipy.misc.imread(file_name))
    if resize:
        image = scipy.misc.imresize(image, size=config.TRAIN.resize, interp='bilinear', mode=None)
    if fliplr:
        image = np.fliplr(image)
    return np.expand_dims(image, axis=0)


def save_image(output, file_name):
    output = np.minimum(np.maximum(output, 0.0), 255.0)
    scipy.misc.toimage(output.squeeze(axis=0), cmin=0, cmax=255).save(file_name)
    return


def save_heat_map(data, path, cur_dir_name, whole_image_name, true_crop_name, resize_dims, resize=False):
    # Optional resize for small images
    if resize:
        data = scipy.misc.imresize(data, size=resize_dims, interp='bicubic', mode=None)

    # Actual saving
    plt.imsave(os.path.join(path, cur_dir_name + '_' + true_crop_name +
                            '_whole_pic_' + whole_image_name + '_heat_map.jpg'), data, cmap='plasma')


def write_loss_in_txt(targetdir, loss_dict, epoch):
    target = open(os.path.join(targetdir, "epoch_%04d_score.txt" % epoch), 'w')
    losses_list = []
    for key, value in loss_dict.items():
        target.write("%s: loss=%f\n" % (key, value))
        losses_list.append(value)
    target.write("\nTotal average loss=%f" % np.mean(losses_list))
    target.close()


def random_crop_together(im1, im2, size):
    images = tf.concat([im1, im2], axis=0)
    images_croped = tf.random_crop(images, size=size)
    im1, im2 = tf.split(images_croped, 2, axis=0)
    return im1, im2


def build_dict():
    val_file_list = []
    data_dict = {}
    true_crop_dict = {}
    total_len = 0

    # validation file list
    val_dir = os.path.join(config.base_dir, config.VAL.A_data_dir)
    v_file_list = [k for k in os.listdir(val_dir) if k.endswith(".jpg")]

    # pick a random true_crop to be the reference
    val_true_crop = "%08d" % np.random.randint(int(v_file_list[-1].split("_")[0])) + "_true_crop.jpg"

    # iterate over all the data folders
    for i in os.listdir(os.path.join(config.base_dir, config.TRAIN.A_data_dir)):
        cur_train_dir = os.path.join(config.base_dir, config.TRAIN.A_data_dir, i)

        # list all the crops .jpg files in the current dir.
        # this includes only the _crop pics not the originals
        file_list = [j for j in os.listdir(cur_train_dir) if (j.endswith(".jpg") and (len(j.split("_")) > 1))]

        # pick a random reference true_crop that will be matched against the others
        true_crop = "%08d" % np.random.randint(int(file_list[-1].split("_")[0])) + "_true_crop.jpg"

        # Shuffle train file list randomly
        file_list = np.random.permutation(file_list)
        assert len(file_list) > 0

        train_file_list = file_list[0::config.TRAIN.every_nth_frame]
        val_file_list = v_file_list[0::config.VAL.every_nth_frame]

        # dictionary with the folder name as key and the list of pics as value
        data_dict = {i: train_file_list}
        # dictionary containing the selected true_crop for each folder
        true_crop_dict = {i: true_crop}
        total_len += len(train_file_list)

    return val_file_list, val_true_crop, data_dict, true_crop_dict, total_len


def inflate_heatmap_to_BigPicSize(OrigHeatmap,CropSize):
    # ----------------------------------------------------------------------------
    # Create new inflated heatmaps with zeros and insert OrigHeatmap at top left.
    # Moves pixel values to middle of crop
    # OrigHeatmap is an np.array
    # CropSize is a tuple
    # ----------------------------------------------------------------------------

    # inflate
    InflatedHeatmap = np.zeros((OrigHeatmap.shape[0]+CropSize[0],OrigHeatmap.shape[1]+CropSize[1]), dtype=float)
    for row in range(OrigHeatmap.shape[0]):
        for col in range(OrigHeatmap.shape[1]):
            InflatedHeatmap[row+int(CropSize[0]/2),col+int(CropSize[1]/2)]=OrigHeatmap[row,col]
    return InflatedHeatmap


def CutoffHeatmap(Heatmap,CutoffStdNum=3):
    # ---------------------------------------------------------
    # Calculate cutoff for heatmap on mean+std*CutoffStdNum
    # Make heatmap "binary" - 255 for above cutoff, 0 if below
    # ---------------------------------------------------------

    Cutoff = np.mean(Heatmap)+CutoffStdNum*np.std(Heatmap)
    Cutoff = np.max(Heatmap)*0.1+np.min(Heatmap)*0.9
    for row in range(Heatmap.shape[0]):
        for col in range(Heatmap.shape[1]):
            if Heatmap[row, col] >= Cutoff:
                Heatmap[row, col] = 0  # Too high loss - want it dark
            else:
                Heatmap[row, col] = 255  # Low loss - bright
    return Heatmap

