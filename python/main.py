__author__ = 'adeb'

import sys
import ConfigParser

import theano
import theano.sandbox.cuda

from dataset import Dataset
import nn
import trainer


if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    if len(sys.argv) == 1:
        config.read('adeb.ini')
    else:
        config.read(str(sys.argv[1]))

    theano.sandbox.cuda.use(config.get('general', 'gpu'))

    ds = Dataset(config)

    # net = nn.Network1(ds.patch_width * ds.patch_width, ds.n_classes)

    batch_size = config.getint('training', 'batch_size')
    net = nn.Network2(ds.patch_width, ds.patch_width * ds.patch_width, ds.n_classes, batch_size)

    t = trainer.Trainer(config, net, ds)
    t.train()
