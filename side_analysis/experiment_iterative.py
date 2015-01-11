__author__ = 'adeb'

from shutil import copy2
import inspect
import PIL
import pickle
from spynet.utils.utilities import analyse_classes
from data_brain_parcellation import DatasetBrainParcellation, DataGeneratorBrain
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
from spynet.utils.utilities import load_config
from data_brain_parcellation import generate_and_save


class ExperimentBrain(Experiment):
    """
    Experiment in which the network is iteratively trained on its misclassified datapoints. kind of boosting.
    """
    def __init__(self, exp_name, data_path):
        Experiment.__init__(self, exp_name, data_path)

    def copy_file_virtual(self):
        copy2(inspect.getfile(inspect.currentframe()), self.path)

    def run(self):
        ###### Create the datasets

        # Testing
        data_cf_test = load_config("cfg_testing_data_creation.py")
        data_cf_test.general["file_path"] = data_cf_test.data_path + "test_temp.h5"
        data_generator_test = DataGeneratorBrain()
        data_generator_test.init_from_config(data_cf_test)
        ds_test_temp = DatasetBrainParcellation()
        vx, inputs, outputs, file_ids = data_generator_test.generate_parallel(data_cf_test.general["n_data"])
        ds_test_temp.populate(inputs, outputs, vx, file_ids)
        ds_test_temp.shuffle_data()

        # Training
        prop_validation = 0.1
        data_cf_train = load_config("cfg_training_data_creation.py")
        data_cf_train.general["file_path"] = data_cf_train.data_path + "train_temp.h5"
        data_generator_train = DataGeneratorBrain()
        data_generator_train.init_from_config(data_cf_train)
        ds_train_temp = DatasetBrainParcellation()
        vx, inputs, outputs, file_ids = data_generator_train.generate_parallel(data_cf_train.general["n_data"])
        ds_train_temp.populate(inputs, outputs, vx, file_ids)
        ds_train_temp.shuffle_data()
        [ds_train_temp, ds_validation_temp] = ds_train_temp.split_dataset_proportions([1-prop_validation, prop_validation])


        ## Scale some part of the data
        print "Scaling"
        s = Scaler([slice(-134, None, None)])
        s.compute_parameters(ds_train_temp.inputs)
        s.scale(ds_train_temp.inputs)
        s.scale(ds_validation_temp.inputs)
        s.scale(ds_test_temp.inputs)
        pickle.dump(s, open(self.path + "s.scaler", "wb"))

        ###### Create the network
        net = NetworkUltimateConv()
        net.init(29, 29, 13, 134, 135)

        print net

        ###### Configure the trainer

        # Cost function
        cost_function = CostNegLL()

        # Learning update
        learning_rate = 0.05
        momentum = 0.5
        lr_update = LearningUpdateGDMomentum(learning_rate, momentum)

        # Create monitors and add them to the trainer
        freq = 1
        freq2 = 0.00001
        err_training = MonitorErrorRate(freq, "Train", ds_train_temp)
        err_testing = MonitorErrorRate(freq, "Test", ds_test_temp)
        err_validation = MonitorErrorRate(freq, "Val", ds_validation_temp)
        # dice_training = MonitorDiceCoefficient(freq, "Train", ds_train_temp, 135)
        dice_testing = MonitorDiceCoefficient(freq, "Test", ds_test_temp, 135)
        # dice_validation = MonitorDiceCoefficient(freq, "Val", ds_validation_temp, 135)

        # Create stopping criteria and add them to the trainer
        max_epoch = MaxEpoch(100)
        #early_stopping = EarlyStopping(err_validation, 10, 0.99, 5)

        # Create the network selector
        params_selector = ParamSelectorBestMonitoredValue(err_validation)

        # Create the trainer object
        batch_size = 200
        t = Trainer(net, cost_function, params_selector, [max_epoch],
                    lr_update, ds_train_temp, batch_size,
                    [err_training, err_validation, err_testing, dice_testing])

        ###### Train the network

        n_iter = 1
        evolution = np.zeros((2, n_iter))

        n_validation_data = ds_validation_temp.n_data

        for i in xrange(n_iter):

            # Train
            t.train()
            net.save_parameters(self.path + "net.net")

            # Test
            print "ERRORS::::"
            out_pred = net.predict(ds_test_temp.inputs, 1000)
            errors = np.argmax(out_pred, axis=1) != np.argmax(ds_test_temp.outputs, axis=1)
            print np.mean(errors)
            evolution[0,i] = np.mean(errors)

            out_pred = net.predict(ds_validation_temp.inputs, 1000)
            errors = np.argmax(out_pred, axis=1) != np.argmax(ds_validation_temp.outputs, axis=1)
            print np.mean(errors)
            evolution[1,i] = np.mean(errors)

            vx_errors = ds_train_temp.vx[errors]
            file_ids_errors = ds_validation_temp.file_ids[errors]
            inputs_errors = ds_validation_temp.inputs[errors]
            outputs_errors = ds_validation_temp.outputs[errors]

            # Update datasets, trainer, monitors
            n_data = int(round(0.9*data_cf_train.general["n_data"]))
            prop_validation = 0.1
            vx, inputs, outputs, file_ids = data_generator_train.generate_parallel(n_data)
            s.scale(inputs)
            split = int(round(n_data*(1-prop_validation)))
            ds_train_temp.populate(np.concatenate([inputs[0:split], inputs_errors], axis=0),
                                   np.concatenate([outputs[0:split], outputs_errors], axis=0),
                                   np.concatenate([vx[0:split], vx_errors], axis=0),
                                   np.concatenate([file_ids[0:split], file_ids_errors], axis=0))
            ds_train_temp.shuffle_data()

            ds_validation_temp.populate(inputs[split:n_data], outputs[split:n_data], vx[split:n_data], file_ids[split:n_data])
            ds_validation_temp.shuffle_data()

            print ds_train_temp.n_data

            t.init()

        print evolution

        ###### Plot the records
        # save_records_plot(self.path, [err_training, err_validation, err_testing], "err", t.n_train_batches, "upper right")
        # save_records_plot(self.path, [dice_testing], "dice", t.n_train_batches, "lower right")

        ###### Save the network

        net.save_parameters(self.path + "net.net")


if __name__ == '__main__':

    exp_name = "test_iter"
    data_path = "./datasets/test_iter_f1/"

    exp = ExperimentBrain(exp_name, data_path)
    exp.run()