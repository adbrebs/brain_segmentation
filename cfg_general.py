__author__ = 'adeb'

import time
from shutil import copy2
import inspect

from utilities import create_directories

# General
folder_name = "essai" + str(int(time.time()))
folder_path = create_directories(folder_name)
net_path = folder_path + "net.net"

# Datasets
create_data = False
data_path = "./data/"
training_data_path = data_path + "train.h5"
testing_data_path = data_path + "test.h5"
if create_data:
    cf_training_file = "cfg_training_data_creation"
    cf_testing_file = "cfg_testing_data_creation"
    copy2(cf_training_file + ".py", folder_path)
    copy2(cf_testing_file + ".py", folder_path)
    cfg_train_data = __import__(cf_training_file)
    cfg_test_data = __import__(cf_testing_file)
    cfg_train = cfg_train_data.CF(training_data_path)
    cfg_test = cfg_test_data.CF(testing_data_path)

prop_validation = 0.3

# Training parameters
batch_size = 200
n_epochs = 1000
patience_increase = 10
improvement_threshold = 0.999
learning_update = "GDmomentum"  # GD, GDmomentum
learning_rate = 0.13
momentum = 0.5

# Copy the config file
copy2(inspect.getfile(inspect.currentframe()), folder_path)