__author__ = 'adeb'

import time
from shutil import copy2
import inspect

from utilities import create_directories

### General
experiment_name = "essai"

### Datasets
create_data = False
if not create_data:
    data_path = "./data/"
    training_data_path = data_path + "ultimate_train.h5"
    testing_data_path = data_path + "ultimate_test.h5"

prop_validation = 0.3

### Training parameters
batch_size = 200
n_epochs = 1000
patience_increase = 10
improvement_threshold = 0.999
learning_update = "GDmomentum"  # GD, GDmomentum
learning_rate = 0.13
momentum = 0.5


####################################################
############# this part should not be modified
####################################################

folder_name = experiment_name + str(int(time.time()))
folder_path = create_directories(folder_name)
net_path = folder_path + "net.net"


# Copy the config file
copy2(inspect.getfile(inspect.currentframe()), folder_path)

# Create the dataset configs if needed
if create_data:
    cf_training_file = "cfg_training_data_creation"
    cf_testing_file = "cfg_testing_data_creation"
    copy2(cf_training_file + ".py", folder_path)
    copy2(cf_testing_file + ".py", folder_path)
    cfg_train = __import__(cf_training_file)
    cfg_test = __import__(cf_testing_file)
    training_data_path = cfg_train.file_path
    testing_data_path = cfg_test.file_path