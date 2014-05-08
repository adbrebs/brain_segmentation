__author__ = 'adeb'

import sys
import time
import math
import numpy as np

import theano
import theano.tensor as T


class Trainer():
    def __init__(self, config, net, ds):
        print '... configure training'

        self.batch_size = config.getint('training', 'batch_size')
        self.learning_rate = config.getfloat('training', 'learning_rate')
        self.n_epochs = config.getint('training', 'n_epochs')

        self.ds = ds

        self.n_train_batches = ds.n_train / self.batch_size
        self.n_valid_batches = ds.n_valid / self.batch_size
        self.n_test_batches = ds.n_test / self.batch_size

        x = T.matrix('x')  # Minibatch input matrix
        y_true = T.matrix('y_true')  # True output of a minibatch

        # Outpurt of the network
        y_pred = net.forward(x, self.batch_size)

        # Cost the trainer is going to minimize
        cost = self.mse_symb(y_pred, y_true)

        # Compute gradients
        params = net.params
        self.grads = T.grad(cost, params)

        updates = []
        for param_i, grad_i in zip(params, self.grads):
            updates.append((param_i, param_i - self.learning_rate * grad_i))

        idx_batch = T.lscalar()
        id1 = idx_batch * self.batch_size
        id2 = (idx_batch + 1) * self.batch_size
        self.test_model = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.test_x[id1:id2], y_true: ds.test_y[id1:id2]})

        self.validate_model = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.valid_x[id1:id2], y_true: ds.valid_y[id1:id2]})

        self.train_model = theano.function(
            inputs=[idx_batch],
            outputs=cost,
            updates=updates,
            givens={x: ds.train_x[id1:id2], y_true: ds.train_y[id1:id2]})

    @staticmethod
    def mse_symb(y_pred, y_true):
        """Return the mean square error
        Args:
            y_pred (theano.tensor.TensorType): output returned by a network
            y_pred (theano.tensor.TensorType): output returned by a network
        """
        return T.mean(T.sum((y_pred - y_true) * (y_pred - y_true), axis=1))

    @staticmethod
    def error_rate_symb(y_pred, y_true):
        """Return the error rate
        Args:
            y_pred (theano.tensor.TensorType): output returned by a network
            y_pred (theano.tensor.TensorType): output returned by a network
        """
        return T.mean(T.neq(T.argmax(y_pred, axis=1), T.argmax(y_true, axis=1)))

    @staticmethod
    def error_rate(y_pred, y_true):
        """Return the error rate
        Args:
            y_pred (theano.tensor.TensorType): output returned by a network
            y_pred (theano.tensor.TensorType): output returned by a network
        """
        return np.mean(np.argmax(y_pred, axis=1) != np.argmax(y_true, axis=1))

    @staticmethod
    def negative_log_likelihood_symb(y_pred, y_true):
        """Return the negative log-likelihood
        Args:
            y_pred (theano.tensor.TensorType): output returned by a network
            y_pred (theano.tensor.TensorType): output returned by a network
        """
        return -T.mean(T.sum(T.log(y_pred) * y_true, axis=1))

    @staticmethod
    def negative_log_likelihood(y_pred, y_true):
        """Return the negative log-likelihood
        Args:
            y_pred (theano.tensor.TensorType): output returned by a network
            y_pred (theano.tensor.TensorType): output returned by a network
        """
        return -np.mean(np.sum(math.log(y_pred) * y_true, axis=1))

    @staticmethod
    def negative_log_likelihood(y_pred, y_true):
        """Return the negative log-likelihood
        Args:
            y_pred (theano.tensor.TensorType): output returned by a network
            y_pred (theano.tensor.TensorType): output returned by a network
        """
        return np.mean(np.sum(math.log(y_pred) * y_true, axis=1))

    def train(self):
        print '... train the network'

        start_time = time.clock()

        # early-stopping parameters
        patience = 5000  # look as this many minibatches regardless
        patience_increase = 1000  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.

        epoch = 0
        early_stopping = False
        id_mini_batch = 0

        while (epoch < self.n_epochs) and (not early_stopping):
            epoch += 1
            for minibatch_index in xrange(self.n_train_batches):

                id_mini_batch += 1

                if id_mini_batch % 100 == 0:
                    print('epoch %i, minibatch %i/%i' % (epoch, minibatch_index + 1, self.n_train_batches))

                self.train_model(minibatch_index)

                if patience <= id_mini_batch:
                    early_stopping = True
                    break

                if (id_mini_batch + 1) % validation_frequency > 0:
                    continue

                # compute validation error
                validation_losses = [self.validate_model(i) for i in xrange(self.n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f' %
                      (epoch, minibatch_index + 1, self.n_train_batches, this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss >= best_validation_loss:
                    continue

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience += patience_increase

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = id_mini_batch

                # test it on the test set
                test_losses = [self.test_model(i) for i in xrange(self.n_test_batches)]
                test_score = np.mean(test_losses)
                print('     epoch %i, minibatch %i/%i, test error of best model %f' %
                      (epoch, minibatch_index + 1, self.n_train_batches, test_score))

        end_time = time.clock()
        print('Training complete.')
        print('Best validation score of %f obtained at iteration %i with test performance %f' %
              (best_validation_loss, best_iter + 1, test_score))
        print >> sys.stderr, ('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))
