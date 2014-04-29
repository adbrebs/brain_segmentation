__author__ = 'adeb'

import ConfigParser

import theano
import theano.sandbox.cuda

from dataset import Dataset
import nn
import trainer


if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    # config.read(str(sys.argv[1]))
    config.read('adeb.ini')
    theano.sandbox.cuda.use(config.get('general', 'gpu'))

    ds = Dataset(config)

    net = nn.Network1(ds.patch_width * ds.patch_width, ds.n_classes)

    t = trainer.Trainer(config, net, ds)
    t.train()
