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

from utilities import analyse_data
import learning_update


class Trainer():
    def __init__(self, config, net, ds, scale=True):
        print '... configure training'

        self.net = net

        analyse_data(ds.train_out.get_value())

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
        lr_update = learning_update.LearningUpdateGDMomentum(self.learning_rate, 0.5)
        updates = lr_update.compute_updates(params, grads)

        idx_batch = T.lscalar()
        id1 = idx_batch * self.batch_size
        id2 = (idx_batch + 1) * self.batch_size
        self.testing_error = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.test_in[id1:id2], y_true: ds.test_out[id1:id2]})

        self.validation_error = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.valid_in[id1:id2], y_true: ds.valid_out[id1:id2]})

        self.training_error = theano.function(
            inputs=[idx_batch],
            outputs=self.error_rate_symb(y_pred, y_true),
            givens={x: ds.train_in[id1:id2], y_true: ds.train_out[id1:id2]})

        self.train_model = theano.function(
            inputs=[idx_batch],
            outputs=cost,
            updates=updates,
            givens={x: ds.train_in[id1:id2], y_true: ds.train_out[id1:id2]})

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
        test_error = 0.
        best_params = None

        freq_display_batch = self.n_train_batches / 4
        epoch = 0
        early_stopping = False
        id_mini_batch = 0

        starting_epoch_time = time.clock()

        training_error_records = []
        testing_error_records = []
        validation_error_records = []

        # Before starting training, evaluate the initial model
        self.__save_record(id_mini_batch, 0, self.training_error,
                           training_error_records, "training error")
        self.__save_record(id_mini_batch, 0, self.validation_error,
                           validation_error_records, "validation error")
        self.__save_record(id_mini_batch, 0, self.testing_error,
                           testing_error_records, "test error of the best model so far")

        while (epoch < self.n_epochs) and (not early_stopping):
            epoch += 1
            print("epoch {}".format(epoch))
            for minibatch_index in xrange(self.n_train_batches):

                id_mini_batch += 1

                # Display minibatch number
                if id_mini_batch % freq_display_batch == 0:
                    print("    minibatch {}/{}".format(minibatch_index + 1, self.n_train_batches))

                # Train on the current minibatch
                self.train_model(minibatch_index)

                # Early stopping
                if patience <= id_mini_batch:
                    early_stopping = True
                    break

                if (id_mini_batch + 1) % validation_frequency > 0:
                    continue

                # Compute the training error and save it
                self.__save_record(id_mini_batch, minibatch_index, self.training_error,
                                   training_error_records, "training error")

                # compute validation error
                valid_error = self.__save_record(id_mini_batch, minibatch_index, self.validation_error,
                                                 validation_error_records, "validation error")

                # if we get the lowest validation error until now
                if valid_error >= best_validation_loss:
                    continue

                #improve patience if loss improvement is good enough
                if valid_error < best_validation_loss * improvement_threshold:
                    patience = id_mini_batch + patience_increase

                # save the lowest validation error
                best_validation_loss = valid_error
                best_iter = id_mini_batch
                best_params = self.net.export_params()

                # test it on the test set
                test_error = self.__save_record(id_mini_batch, minibatch_index, self.testing_error,
                                                testing_error_records, "test error of the best model so far")

            print("    epoch {} finished after {} seconds".format(epoch, time.clock() - starting_epoch_time))
            starting_epoch_time = time.clock()

        end_time = time.clock()
        self.net.import_params(best_params)
        print("Training complete.")
        print('Best validation error of {} obtained at iteration {} with test performance {}'.format(
            best_validation_loss, best_iter + 1, test_error))
        print >> sys.stderr, ("Training ran for {} minutes".format((end_time - start_time) / 60.))

        self.__save_records("t.png", training_error_records, testing_error_records, validation_error_records)

    def __save_record(self, id_mini_batch, minibatch_index, error_function, error_records, name):
        losses = [error_function(i) for i in xrange(self.n_test_batches)]
        error = np.mean(losses)
        print("    minibatch {}/{}, {}: {}".format(minibatch_index + 1, self.n_train_batches, error, name))
        error_records.append((id_mini_batch, error))
        return error

    @staticmethod
    def __save_records(file_name, training_error_records, testing_erro_records, validation_error_records):

        def save_error(error, legend):
            plt.plot(*zip(*error), label=legend)

        save_error(training_error_records, "training data")
        save_error(testing_erro_records, "testing data")
        save_error(validation_error_records, "validation data")

        plt.xlabel('Minibatch index')
        plt.ylabel('Error rate')
        plt.legend(loc='upper right')
        plt.savefig('./images/training/' + file_name)