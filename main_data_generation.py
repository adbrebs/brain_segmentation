__author__ = 'adeb'

from utilities import load_config
from dataset import DatasetBrainParcellation


if __name__ == '__main__':

    ### Load config
    cfg = load_config("cfg_training_data_creation")

    ### File name
    file_name = cfg.get('generate_data', 'file_name')

    # Create the data generator
    dc_training = DatasetBrainParcellation()
    dc_training.populate_from_config(cfg)
    dc_training.write(file_name)