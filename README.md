Retraining VGG For DDIS*

Project A, by George Pisha and Eyal Naor

Background

This code is for retraining VGG19 for template matching with the Deformable Diversity Similarity (DDIS) algorithm.
For the DDIS GitHub page: https://github.com/roimehrez/DDIS
The code in this page was the starting point of this project.

This code enables to retrain VGG19, as pretrained for classification on ImageNet, in order to extract feature vectors that are optimized for template matching with the DDIS algoritm.
It contains the code needed for:
-Preprocessing training data in the format of the Visual Object Tracking (VOT) challenges
-Training the network
-Producing similarity heatmaps for the test batch

The Report.pdf depicts in detail the process, as well as shows examples of output.

Use

Training and Testing Material
-The training material can be downloaded from http://www.votchallenge.net/vot2017/dataset.html 
-image_crop.py is used to create user determined crops from the data, used for training.
	The file is modified to wirk on VOT2016 data, with its labels, names, etc. For other data sources, adjustments need to be made.
	In the crop.py file are explanations on the different configuration settings, such as input/output folder, crop characteristics, etc.
-Division of data to Training/Validation/Testing can be done with make_train_val_test_dirs.py ,with user determined ratios

Configuration
The main configuration parameters are depicted and user determined in config.py.
The main parameters that need to be adjusted to user's configuration and wanted training type:
- config.base_dir : Base location for the project
- config.TRAIN.A_data_dir : Relative location from base_dir of the training data directory, as created with image_crop.py
- config.TRAIN.out_dir : Relative location from base_dir of Results folder location, in which the models and logs are saved
- config.vgg_model_path : Location of the pretrained VGG19 network, available for download on this page.
- config.TRAIN.is_train : True for Training, False for Testing
- config.TRAIN.num_epochs : Number of epochs to run while training
- config.TRAIN.save_every_nth_epoch : Determines number of saved models. Each model is ~350MB.
- config.TRAIN.every_nth_frame : Determines number of crops trained on in each data folder
- config.TEST.A_data_dir : Relative location from base_dir of the test data directory
- config.CX.feat_layers : Layers from network that are extracted as the feature vector

Additional parameters are in the main train_vgg_for_cx.py file:
- lr: 1e-8  : Determines the Learning Rate used for training
- G_loss : Determines the Loss Function for training. The two main options used in this project are in the code, where one is hidden.
