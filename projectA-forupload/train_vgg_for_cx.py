# ---------------------------------------------------
#   code credits: https://github.com/CQFIO/PhotographicImageSynthesis
# ---------------------------------------------------
from __future__ import division
import time
import utils.helper as helper
from CX.CX_helper import *
from model import *
from utils.FetchManager import *
import imageio
import scipy.misc
import matplotlib.pyplot as plt


sess = tf.Session()

# ---------------------------------------------------
#                      graph
# ---------------------------------------------------
# set a new scope (name-spacing) for variables
# it's used to be able to share the variables with many process or hardware in training
with tf.variable_scope(tf.get_variable_scope()):
    # Template image, deform image
    # Create an input variable nXmX3 tensor
    in_A = tf.placeholder(tf.float32, [1, None, None, 3])
    in_B = tf.placeholder(tf.float32, [1, None, None, 3])
    input_A = tf.image.resize_bilinear(in_A, (config.TRAIN.sp, config.TRAIN.sp))
    input_B = tf.image.resize_bilinear(in_B, (config.TRAIN.sp, config.TRAIN.sp))

    # Amount of overlap between the image_B and the ground_truth given in VOT
    overlap_input = tf.placeholder(tf.float32)

    # Weights of VGG - stored in the folder as .mat/.npy
    weight = tf.placeholder(tf.float32)

    # Define the nets scope ***WHY NEW SCOPE??***
    with tf.variable_scope("VGG") as scope:
        # Get features for each image
        vgg_B = build_vgg19(input_B)
        vgg_A = build_vgg19(input_A, reuse=True)

    # Resize feature tensor
    def resize_and_concat_features(feat, layers):
        list = []
        sp = feat[layers[0]].shape
        for layer_name in layers:
            feat_resize = tf.image.resize_bilinear(
                        feat[layer_name], (sp[1].value, sp[2].value)) / feat[layer_name].shape[3].value
            list.append(feat_resize)
        stac_feat = tf.concat(list, axis=3)
        return stac_feat

    # MAYBE CHANGE TO ONE LOSS ON THE CONCAT OF FEATURES
    # INSTEAD OF MANY LOSSES ON MANY LAYERS
    # --- contextual e---
    if config.W.CX_content > 0:
        CX_loss_content_list = [w * CX_loss_helper(vgg_A[layer], vgg_B[layer], config.CX)
                                for layer, w in config.CX.feat_content_layers.items()]
        CX_content_loss = tf.reduce_sum(CX_loss_content_list)
        CX_content_loss *= config.W.CX_content
    else:
        CX_content_loss = zero_tensor

    # --- total loss ---
    # TEMP LOSS DEFINITION CHANGE WHEN WE DECIDE WHAT WORKS BEST
    # overlap_input is [1,0], -log to bring to [0,infty], to match DDIS, 0->Identical,infty->Different
    # G_loss = tf.reduce_mean((CX_content_loss + tf.log(overlap_input + config.TRAIN.epsilon)) ** 2)
    # G_loss = CX_content_loss
    tan_overlap = ((np.pi/2)+tf.atan(100*(overlap_input-1/2)))/np.pi
    G_loss = tf.reduce_mean(-(overlap_input)*tf.log(tf.clip_by_value(CX_content_loss, 1e-15, 1.0)) - (1-overlap_input)*tf.log(1-tf.clip_by_value(CX_content_loss, 1e-15, 1.0)))

# create the optimization
# Define all the hyper-params and optimizer
lr = tf.placeholder(tf.float32)
var_list = [var for var in tf.trainable_variables() if var.name.startswith("VGG")]
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=var_list)
saver = tf.train.Saver(max_to_keep=1000)
# initialize the variables of tf
sess.run(tf.global_variables_initializer())


# ----------------------------------------------------------
#               LOAD FUNCTION
# Load TF checkpoint from given path
# If not found continue with original VGG weights
# ----------------------------------------------------------
def load(dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    if ckpt:  # makes sure model exists in dir
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    return ckpt


# ---------------------------------------------------
#                      train
# ---------------------------------------------------
if config.TRAIN.is_train:

    dir_list = os.listdir(os.path.join(config.base_dir, config.TRAIN.A_data_dir))
    val_parent_list = os.path.join(config.base_dir, config.VAL.A_data_dir)
    val_dir_list = os.listdir(val_parent_list)
    result_dir = os.path.join(config.base_dir, config.TRAIN.out_dir)

    # instantiate a FetchManager class
    fetcher = FetchManager(sess, [G_opt, G_loss])

    # -------------------- epoch loop -------------------------
    #
    #   Will contain all the data needed to iterate
    #   over all the folders in the data set
    #
    # -------------------- epoch loop -------------------------
    for epoch in range(1, config.TRAIN.num_epochs + 1):
        # a folder to save the epoch results
        epoch_dir = os.path.join(result_dir, "%04d" % epoch)

        if os.path.isdir(epoch_dir):
            continue
        cnt = 0

        Epoch_g_loss = {}

        # ------------ batch loop -------------------------
        for dirIndex in np.random.permutation(len(dir_list)):
            cur_dir = dir_list[dirIndex]

            cur_train_dir = os.path.join(config.base_dir, config.TRAIN.A_data_dir, cur_dir)

            # list all the crops .jpg files in the current dir.
            # this includes only the _crop pics not the originals or true_crops
            # file_name format: framenum(8 digit long int)_crop_cropnum_overlap(0-100)
            """When blackboxing - this part will need to generate crops from original frames"""
            file_list = [str(j).split(".")[0] for j in os.listdir(cur_train_dir) if (j.endswith(".jpg") and (len(j.split("_")) > 3))]

            g_loss = np.zeros(len(file_list), dtype=float)
            cx_loss = np.zeros(len(file_list), dtype=float)

            # pick a random reference true_crop that will be matched against the others
            true_crop = "%08d" % np.random.randint(int(file_list[-1].split("_")[0])) + "_true_crop"

            # Shuffle training file list randomly
            file_list = np.random.permutation(file_list)
            assert len(file_list) > 0

            train_file_list = file_list[0::config.TRAIN.every_nth_frame]

            for ind in np.random.permutation(len(train_file_list)):
                st = time.time()
                cnt += 1

                A_file_name = os.path.join(config.base_dir, config.TRAIN.A_data_dir, cur_dir, train_file_list[ind]+'.jpg')
                B_file_name = os.path.join(config.base_dir, config.TRAIN.A_data_dir, cur_dir, true_crop+'.jpg')

                # last parameter in file_name, normalized to [0,1]
                overlap = float(train_file_list[ind].split("_")[-1])/100

                if not os.path.isfile(A_file_name) or not os.path.isfile(B_file_name):
                    continue
                A_image = helper.read_image(A_file_name)  # training image A
                B_image = helper.read_image(B_file_name)  # training image B

                feed_dict = {input_A: A_image, input_B: B_image, overlap_input: overlap, lr: 1e-8}

                # session run
                eval = fetcher.fetch(feed_dict, [G_loss])
                # evalCX = fetcher.fetch(feed_dict, [CX_content_loss])
                if cnt % 100 == 0:
                    g_loss[ind] = eval[G_loss]
                    # cx_loss[ind] = evalCX[CX_content_loss]
                    log = "epoch:%d | cnt:%d | time:%.2f | loss:%.2f ||  cur_dir: %s"\
                          % (epoch, cnt, (time.time() - st), float(np.mean(g_loss[np.where(g_loss)])), cur_dir)
                    print(log)

            # ------------ end batch loop -------------------
            # Append curDirloss to total Epoch_loss of current epoch
            Epoch_g_loss[cur_dir] = np.mean(g_loss[np.where(g_loss)])

        # -------------- save the model ------------------
        # save only every nth epoch (save space)
        # we use loop with try and catch to verify that the save was done.
        # when saving on Dropbox it sometimes cause an error.
        for i in range(5):
            try:
                TrainLogFolder = os.path.join(result_dir, "TrainLogs")
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                if not os.path.exists(TrainLogFolder):
                    os.makedirs(TrainLogFolder)
                # save the mean of the epochs train loss
                helper.write_loss_in_txt(TrainLogFolder, Epoch_g_loss, epoch)
                if (epoch % config.TRAIN.save_every_nth_epoch) == 0:
                    saver.save(sess, os.path.join(epoch_dir, "model.ckpt"))
            except:
                time.sleep(1)

        # ------------ validation loop -------------------------
        Val_losses_dict = {}
        for ValDirIndex in np.random.permutation(len(val_dir_list)):
            cur_val_dir = val_dir_list[ValDirIndex]  # folder name
            cur_val_dir_path = os.path.join(val_parent_list, cur_val_dir)  # total path to folder

            v_file_list = [str(j).split(".")[0] for j in os.listdir(cur_val_dir_path) if (j.endswith(".jpg") and (len(j.split("_")) > 3))]

            val_file_list = v_file_list[0::config.VAL.every_nth_frame]

            curDEBUG = val_file_list[-1]
            val_g_loss = np.zeros(len(val_file_list), dtype=float)
            val_log_counter = 0
            for ind in range(1, len(val_file_list)):

                true_crop = "%08d" % np.random.randint(1, int(val_file_list[-1].split("_")[0])) + "_true_crop"
                A_file_name = os.path.join(cur_val_dir_path, val_file_list[ind] + '.jpg')
                B_file_name = os.path.join(cur_val_dir_path, true_crop + '.jpg')

                if not os.path.isfile(A_file_name):  # test label
                    continue
                A_image_val = helper.read_image(A_file_name)  # training image A
                B_image_val = helper.read_image(B_file_name)  # training image A
                overlap_val = float(val_file_list[ind].split("_")[-1])/100
                output = sess.run(G_loss, feed_dict={input_A: A_image_val, input_B: B_image_val, overlap_input: overlap_val})  # overlap_val....
                val_g_loss[ind] = output
                if (val_log_counter % 100) == 0:
                    log = "VAL | folder: %s | epoch:%d | loss:%.2f" % (cur_val_dir, epoch,  float(np.mean(val_g_loss[np.where(val_g_loss)])))
                    val_log_counter += 1
                    print(log)
            # Finished current validation folder - add to log
            Val_losses_dict[cur_val_dir] = np.mean(val_g_loss[np.where(val_g_loss)])
        # finished all validation folders - create log file
        for i in range(5):
            try:
                ValLogFolder = os.path.join(result_dir, "ValidationLogs")
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                if not os.path.exists(ValLogFolder):
                    os.makedirs(ValLogFolder)
                # save the mean of the epochs val loss
                helper.write_loss_in_txt(ValLogFolder, Val_losses_dict, epoch)
            except:
                time.sleep(1)


# ---------------------------------------------------
#                      test
# ---------------------------------------------------
if config.TEST.is_test:
    for model in range(0, config.TRAIN.num_epochs + 1, config.TRAIN.save_every_nth_epoch):
        # load the model from the dir defined in config.py
        # calls load above
        ckpt = load(os.path.join(config.base_dir, config.TRAIN.out_dir, "%04d" % model))
        print(ckpt)
        test_dir_path = os.path.join(config.base_dir, config.TEST.A_data_dir)
        test_dir_list = os.listdir(test_dir_path)

        for TestDirIndex in np.random.permutation(len(test_dir_list)):
            cur_test_dir = test_dir_list[TestDirIndex]
            cur_test_path = os.path.join(test_dir_path, cur_test_dir)

            # list all the files in the folder
            file_list = [str(j).split(".")[0] for j in os.listdir(cur_test_path) if (j.endswith(".jpg") and (len(j.split("_")) > 3))]

            # randomly pick a true crop to compare the crops against
            #true_crop = "%08d" % np.random.randint(1, int(file_list[-1].split("_")[0])) + "_true_crop"
            true_crop = "%08d" % int(len(file_list)/77) + "_true_crop"
            #true_crop = "%08d" % int(94) + "_true_crop" #291018
            true_crop_path = os.path.join(cur_test_path, true_crop + '.jpg')

            # randomly pick a whole image to slice and compare silces against true crop above
            whole_image = "%08d" % np.random.randint(1, int(file_list[-1].split("_")[0]))
            whole_image = "%08d" % int(len(file_list) / 77)
            #whole_image = "%08d" % int(94) #291018
            whole_image_path = os.path.join(cur_test_path, whole_image + '.jpg')

            # open true_crop as np array
            TruthPic = np.float32(imageio.imread(true_crop_path))
            # find the h,w of true_crop
            TruthSize = TruthPic.shape

            # load the full frame
            BigPic = np.float32(imageio.imread(whole_image_path))
            # find the shape of the frame (used to determine the boarders of the crops)
            BigSize = BigPic.shape

            # Resize the true_crop of the image so it fits the input-size of the net
            if config.TEST.is_resize:
                TruthPic = scipy.misc.imresize(TruthPic, size=config.TRAIN.resize, interp='bilinear', mode=None)
            if config.TEST.is_fliplr:
                TruthPic = np.fliplr(TruthPic)

            # make size (1,width,height,rgb) to fit net input size
            TruthPic = np.expand_dims(TruthPic, axis=0)

            # Matrix to hold losses of each relevant crop
            heat_map_size = [BigSize[0] - TruthSize[0], BigSize[1] - TruthSize[1]]
            # Since we want entire crop to be inside orig image, matrix is smaller
            test_g_loss = np.zeros((len(range(0, heat_map_size[0], config.TEST.jump_size)),
                                    len(range(0, heat_map_size[1], config.TEST.jump_size))), dtype=float)

            counter = 0
            for row in range(0, heat_map_size[0], config.TEST.jump_size):
                for col in range(0, heat_map_size[1], config.TEST.jump_size):
                    pct_finished = 100*(counter / (test_g_loss.shape[0]*test_g_loss.shape[1]))
                    print("Folder: ", cur_test_dir, " | ", "%.2f" % pct_finished, "%")
                    CurCrop = BigPic[row:row + TruthSize[0], col:col + TruthSize[1]]
                    # print(CurCrop.shape)
                    if config.TEST.is_resize:
                        CurCrop = scipy.misc.imresize(CurCrop, size=config.TRAIN.resize, interp='bilinear', mode=None)
                    if config.TEST.is_fliplr:
                        CurCrop = np.fliplr(CurCrop)

                    # make size (1,width,height,rgb) to fit net input size
                    CurCrop = np.expand_dims(CurCrop, axis=0)
                    # We set overlap_input such that the loss for curCrop will be the actual loss
                    # and not distance from our expected loss, as it is while training.
                    output = sess.run(CX_content_loss, feed_dict={input_A: CurCrop, input_B: TruthPic, overlap_input: 1})
                    test_g_loss[int(row/config.TEST.jump_size)][int(col/config.TEST.jump_size)] = output
                    counter += 1

            test_g_loss = scipy.misc.imresize(test_g_loss, size=tuple(heat_map_size), interp='bicubic', mode=None)
            # test_g_loss = helper.CutoffHeatmap(test_g_loss, 3)
            test_g_loss = helper.inflate_heatmap_to_BigPicSize(test_g_loss, TruthSize)
            if model > 0:
                plt.imsave(os.path.join(config.base_dir, "model_" + str(model) + '_' + cur_test_dir + '_' + true_crop +
                                        '_whole_pic_' + whole_image + '_heat_map.jpg'), test_g_loss, cmap='plasma')
            else:
                plt.imsave(os.path.join(config.base_dir, 'VGG' + '_' + cur_test_dir + '_' + true_crop +
                                        '_whole_pic_' + whole_image + '_heat_map.jpg'), test_g_loss, cmap='plasma')