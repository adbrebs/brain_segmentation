__author__ = 'adeb'

import sys
import ConfigParser

import theano
import theano.sandbox.cuda

from database import DataBase
import nn
import trainer


def load_config():
    cf = ConfigParser.ConfigParser()
    if len(sys.argv) == 1:
        cf.read('training.ini')
    else:
        cf.read(str(sys.argv[1]))
    theano.sandbox.cuda.use(cf.get('general', 'gpu'))
    return cf

if __name__ == '__main__':

    ### Load the config file
    training_cf = load_config()

    ### Create the database
    ds = DataBase(training_cf)

    ### Create the network
    # MLP kind network
    # net = nn.Network1(ds.patch_width * ds.patch_width, ds.n_classes)

    # CNN network
    net = nn.Network3(ds.patch_width, ds.n_classes)

    # ### Train the network
    t = trainer.Trainer(training_cf, net, ds)
    t.train()

    ### Save the network
    net.save_parameters("net2.net")