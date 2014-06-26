__author__ = 'adeb'

from shutil import copy2
import inspect

from spynet.utils.utilities import create_directories
from spynet.models.network import lookup_network


############################################################
############# this part can be modified ####################
############################################################

### General
experiment_name = "allez_la"  # Folder that will contain the results of an experiment

### Datasets
data_path = "./datasets/ultimate/"
training_data_path = data_path + "train.h5"
testing_data_path = data_path + "test.h5"
prop_validation = 0.3  # Percentage of the training dataset that is used for validation (early stopping)

### Training parameters
batch_size = 200
n_epochs = 1000
patience_increase = 10
improvement_threshold = 0.995
learning_update = "GDmomentum"  # GD, GDmomentum
learning_rate = 0.13
momentum = 0.5


############################################################
############# this part should not be modified #############
############################################################

experiment_path = create_directories(experiment_name)
net_path = experiment_path + "net.net"

# Copy the config file into the experiment path
copy2(inspect.getfile(inspect.currentframe()), experiment_path)