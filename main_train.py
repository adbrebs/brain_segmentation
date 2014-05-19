__author__ = 'adeb'

import sys
import ConfigParser

from database import DataBaseBrainParcellation
import nn
from trainer import Trainer


def load_config():
    cf = ConfigParser.ConfigParser()
    if len(sys.argv) == 1:
        cf.read('training.ini')
    else:
        cf.read(str(sys.argv[1]))
    return cf

if __name__ == '__main__':

    ### Load the config file
    training_cf = load_config()

    ### Create the database
    db = DataBaseBrainParcellation()
    db.load_from_config(training_cf)

    ### Create the network
    # MLP kind network
    # net = nn.Network1(ds.patch_width * ds.patch_width, ds.n_classes)
    # CNN network
    net = nn.Network2()
    net.init(db.patch_width, db.n_out_features)

    ### Train the network
    t = Trainer(training_cf, net, db)
    t.train()

    ### Save the network
    net.save_parameters(training_cf.get("general", "net"))