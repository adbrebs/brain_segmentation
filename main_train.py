__author__ = 'adeb'

from utilities import load_config, create_directories
from database import DataBaseBrainParcellation
from dataset import generate_and_save
import network as nn
from trainer import Trainer


if __name__ == '__main__':

    ### Load the config file
    training_cf = load_config("cfg_general")

    ### Create datasets if specified
    if training_cf.create_data:
        generate_and_save(training_cf.cfg_train)
        generate_and_save(training_cf.cfg_test)

    ### Create the database
    db = DataBaseBrainParcellation()
    db.init_from_config(training_cf)

    ### Create the network
    # MLP kind network
    net = nn.Network1()
    net.init(db.patch_width*db.patch_width, db.n_out_features)
    # CNN network
    # net = nn.Network2()
    # net.init(db.patch_width, db.patch_width, db.n_out_features)
    # net = nn.Network3()
    # net.init(db.patch_width, db.patch_width, db.patch_width, db.n_out_features)
    print net

    ### Train the network
    t = Trainer(training_cf, net, db)
    t.train()

    ### Save the network
    net.save_parameters(training_cf.net_path)