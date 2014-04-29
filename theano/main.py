__author__ = 'adeb'

import ConfigParser

import theano
import theano.sandbox.cuda
import theano.tensor as T

from dataset import Dataset
import nn
import trainer


if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    # config.read(str(sys.argv[1]))
    config.read('adeb.ini')
    training_data = config.get('general', 'training_data')
    testing_data = config.get('general', 'testing_data')
    theano.sandbox.cuda.use(config.get('general', 'gpu'))
    batch_size = config.get('general', 'batch_size')
    learning_rate = config.get('general', 'learning_rate')

    ds = Dataset(training_data, testing_data)
    x = T.matrix('x')

    net = nn.Network1(ds.patch_width * ds.patch_width, ds.n_classes, x)

    t = trainer.Trainer(net, ds)
    t.train()
