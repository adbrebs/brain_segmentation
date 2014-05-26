__author__ = 'adeb'

import sys
import time
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import theano
import theano.tensor as T

from dataset import analyse_data
import learning_update


class Trainer():
    def __init__(self, config, net, ds, scale=True):
        print '... configure training'

        self.net = net

        analyse_data(ds.train_y.get_value())

        # Scale the data
        if scale:
            net.create_scaling_from_raw_database(ds)
            net.scale_database(ds)

        self.patience_increase = config.getint('training', 'patience_increase')
        self.improvement_threshold = config.getfloat('training', 'improvement_threshold')
        self.batch_size = config.getint('training', 'batch_size')
        self.learning_rate = config.getfloat('training', 'learning_rate')
        self.n_epochs = config.getint('training', 'n_epochs')

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
        grads = T.grad(cost, params)

        # Compute updates
        lr_update = learning_update.LearningUpdateGDMomentum(self.learning_rate, 0.9)
        updates = lr_update.compute_updates(params, grads)

        idx_batch = T.lscalar()
        id1 = idx_batch * self.batch_size
        id2 = (idx_batch + 1) * self.batch_size
        self.testing_score = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.test_x[id1:id2], y_true: ds.test_y[id1:id2]})

        self.validation_score = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.valid_x[id1:id2], y_true: ds.valid_y[id1:id2]})

        self.training_score = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.train_x[id1:id2], y_true: ds.train_y[id1:id2]})

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
        patience = 10 * self.n_train_batches  # look as this many minibatches regardless
        patience_increase = self.patience_increase * self.n_train_batches
        improvement_threshold = self.improvement_threshold
        validation_frequency = min(self.n_train_batches, patience / 2)

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        best_params = None

        freq_display_batch = self.n_train_batches / 4
        epoch = 0
        early_stopping = False
        id_mini_batch = 0

        starting_epoch_time = time.clock()

        training_score_records = []
        testing_score_records = []
        validation_score_records = []

        while (epoch < self.n_epochs) and (not early_stopping):
            epoch += 1
            print("epoch {}".format(epoch))
            for minibatch_index in xrange(self.n_train_batches):

                id_mini_batch += 1

                if id_mini_batch % freq_display_batch == 0:
                    print("    minibatch {}/{}".format(minibatch_index + 1, self.n_train_batches))

                self.train_model(minibatch_index)

                if patience <= id_mini_batch:
                    early_stopping = True
                    break

                if (id_mini_batch + 1) % validation_frequency > 0:
                    continue

                # compute training error
                training_losses = [self.training_score(i) for i in xrange(self.n_valid_batches)]
                this_training_loss = np.mean(training_losses)
                print("    minibatch {}/{}, training error: {}".format(
                    minibatch_index + 1, self.n_train_batches, this_training_loss))
                training_score_records.append((id_mini_batch, this_training_loss))

                # compute validation error
                validation_losses = [self.validation_score(i) for i in xrange(self.n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print("    minibatch {}/{}, validation error: {}".format(
                    minibatch_index + 1, self.n_train_batches, this_validation_loss))
                validation_score_records.append((id_mini_batch, this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss >= best_validation_loss:
                    continue

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = id_mini_batch + patience_increase

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = id_mini_batch
                best_params = self.net.export_params()

                # test it on the test set
                test_losses = [self.testing_score(i) for i in xrange(self.n_test_batches)]
                test_score = np.mean(test_losses)
                print("    minibatch {}/{}, test error of the best model so far: {}".format(
                    minibatch_index + 1, self.n_train_batches, test_score))
                testing_score_records.append((id_mini_batch, test_score))

            print("    epoch {} finished after {} seconds".format(epoch, time.clock() - starting_epoch_time))
            starting_epoch_time = time.clock()

        end_time = time.clock()
        self.net.import_params(best_params)
        print('Training complete.')
        print('Best validation score of %f obtained at iteration %i with test performance %f' %
              (best_validation_loss, best_iter + 1, test_score))
        print >> sys.stderr, ('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))

        self.__save_records("t.png", training_score_records, testing_score_records, validation_score_records)

    @staticmethod
    def __save_records(file_name, training_score_records, testing_score_records, validation_score_records):

        def save_score(score, legend):
            plt.plot(*zip(*score), label=legend)

        save_score(training_score_records, "training data")
        save_score(testing_score_records, "testing data")
        save_score(validation_score_records, "validation data")

        plt.xlabel('Minibatch index')
        plt.ylabel('Error rate')
        plt.legend(loc='upper right')
        plt.savefig('./images/training/' + file_name)
