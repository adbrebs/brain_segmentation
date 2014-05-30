__author__ = 'adeb'

from utilities import load_config
from dataset import DatasetBrainParcellation


if __name__ == '__main__':

    ### Load config
    config_ini = load_config("creation_training_data_1.ini")

    ### File name
    file_name = config_ini.get('generate_data', 'file_name')

    # Create the data generator
    dc_training = DatasetBrainParcellation()
    dc_training.populate_from_config(config_ini)
    dc_training.write(file_name)