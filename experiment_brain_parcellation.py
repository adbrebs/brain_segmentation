__author__ = 'adeb'

from shutil import copy2
import inspect
import PIL
import pickle
from spynet.utils.utilities import analyse_classes
from data_brain_parcellation import DatasetBrainParcellation
from network_brain_parcellation import *
from spynet.models.network import *
from spynet.models.neuron_type import *
from spynet.data.dataset import *
from spynet.training.trainer import *
from spynet.training.monitor import *
from spynet.training.parameters_selector import *
from spynet.training.stopping_criterion import *
from spynet.training.cost_function import *
from spynet.training.learning_update import *
from spynet.experiment import Experiment
from spynet.utils.utilities import tile_raster_images
import theano


class ExperimentBrain(Experiment):
    """
    Main experiment to train a network on a dataset
    """
    def __init__(self, exp_name, data_path):
        Experiment.__init__(self, exp_name, data_path)

    def copy_file_virtual(self):
        copy2(inspect.getfile(inspect.currentframe()), self.path)

    def run(self):
        ###### Create the datasets

        # aa = CostNegLLWeighted(np.array([0.9, 0.1]))
        # e = theano.function(inputs=[], outputs=aa.test())
        # print e()

        ## Load the data
        training_data_path = self.data_path + "train.h5"
        ds_training = DatasetBrainParcellation()
        ds_training.read(training_data_path)

        [ds_training, ds_validation] = ds_training.split_dataset_proportions([0.95, 0.05])

        testing_data_path = self.data_path + "test.h5"
        ds_testing = DatasetBrainParcellation()
        ds_testing.read(testing_data_path)

        ## Display data sample
        # image = PIL.Image.fromarray(tile_raster_images(X=ds_training.inputs[0:50],
        #                                                img_shape=(29, 29), tile_shape=(5, 10),
        #                                                tile_spacing=(1, 1)))
        # image.save(self.path + "filters_corruption_30.png")

        ## Few stats about the targets
        classes, proportion_class = analyse_classes(np.argmax(ds_training.outputs, axis=1), "Training data:")
        print classes
        ## Scale some part of the data
        print "Scaling"
        s = Scaler([slice(-134, None, None)])
        s.compute_parameters(ds_training.inputs)
        s.scale(ds_training.inputs)
        s.scale(ds_validation.inputs)
        s.scale(ds_testing.inputs)
        pickle.dump(s, open(self.path + "s.scaler", "wb"))

        ###### Create the network
        net = NetworkUltimateConv()
        net.init(33, 29, 5, 134, 135)

        print net

        ###### Configure the trainer

        # Cost function
        cost_function = CostNegLL(net.ls_params)

        # Learning update
        learning_rate = 0.05
        momentum = 0.5
        lr_update = LearningUpdateGDMomentum(learning_rate, momentum)

        # Create monitors and add them to the trainer
        freq = 1
        freq2 = 0.00001
        # err_training = MonitorErrorRate(freq, "Train", ds_training)
        # err_testing = MonitorErrorRate(freq, "Test", ds_testing)
        err_validation = MonitorErrorRate(freq, "Val", ds_validation)
        # dice_training = MonitorDiceCoefficient(freq, "Train", ds_training, 135)
        dice_testing = MonitorDiceCoefficient(freq, "Test", ds_testing, 135)
        # dice_validation = MonitorDiceCoefficient(freq, "Val", ds_validation, 135)

        # Create stopping criteria and add them to the trainer
        max_epoch = MaxEpoch(300)
        early_stopping = EarlyStopping(err_validation, 10, 0.99, 5)

        # Create the network selector
        params_selector = ParamSelectorBestMonitoredValue(err_validation)

        # Create the trainer object
        batch_size = 200
        t = Trainer(net, cost_function, params_selector, [max_epoch, early_stopping],
                    lr_update, ds_training, batch_size,
                    [err_validation, dice_testing])

        ###### Train the network

        t.train()

        ###### Plot the records

        # pred = np.argmax(t.net.predict(ds_testing.inputs, 10000), axis=1)
        # d = compute_dice(pred, np.argmax(ds_testing.outputs, axis=1), 134)
        # print "Dice test: {}".format(np.mean(d))
        # print "Error rate test: {}".format(error_rate(np.argmax(ds_testing.outputs, axis=1), pred))

        save_records_plot(self.path, [err_validation], "err", t.n_train_batches, "upper right")
        # save_records_plot(self.path, [dice_testing], "dice", t.n_train_batches, "lower right")

        ###### Save the network

        net.save_parameters(self.path + "net.net")


if __name__ == '__main__':

    exp_name = "paper_ultimate_conv"
    data_path = "./datasets/paper_ultimate_conv/"

    exp = ExperimentBrain(exp_name, data_path)
    exp.run()