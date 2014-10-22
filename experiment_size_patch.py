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

from spynet.utils.utilities import load_config
from data_brain_parcellation import generate_and_save


class ExperimentBrain(Experiment):
    def __init__(self, exp_name):
        Experiment.__init__(self, exp_name)

    def copy_file_virtual(self):
        copy2(inspect.getfile(inspect.currentframe()), self.path)

    def run(self):

        data_path = "./datasets/final_exp_size_patch/"
        range_patch_size = np.arange(3, 37, 2)
        error_rates = np.zeros(range_patch_size.shape)
        dice_coeffs = np.zeros(range_patch_size.shape)

        for idx, patch_size in enumerate(range_patch_size):

            print "patch width {}".format(patch_size)

            ### Load the config file
            data_cf_train = load_config("cfg_training_data_creation.py")
            data_cf_test = load_config("cfg_testing_data_creation.py")
            data_cf_train.data_path = data_path
            data_cf_test.data_path = data_path
            data_cf_train.pick_features[0]["patch_width"] = patch_size
            data_cf_test.pick_features[0]["patch_width"] = patch_size

            ### Generate and write on file the dataset
            generate_and_save(data_cf_train)
            generate_and_save(data_cf_test)

            ###### Create the datasets

            training_data_path = data_path + "train.h5"
            testing_data_path = data_path + "test.h5"
            ds_training = DatasetBrainParcellation()
            ds_training.read(training_data_path)
            # image = PIL.Image.fromarray(tile_raster_images(X=ds_training.inputs[0:100],
            #                                                img_shape=(29, 29), tile_shape=(10, 10),
            #                                                tile_spacing=(1, 1)))
            # image.save(self.path + "filters_corruption_30.png")

            ds_validation = DatasetBrainParcellation()
            ds_validation.read(testing_data_path)
            # Few stats about the targets
            analyse_classes(np.argmax(ds_training.outputs, axis=1))

            # Scale some part of the data
            # s = Scaler([slice(-134, None, None)])
            # s.compute_parameters(ds_training.inputs)
            # s.scale(ds_training.inputs)
            # s.scale(ds_validation.inputs)
            # s.scale(ds_testing.inputs)
            # pickle.dump(s, open(self.path + "s.scaler", "wb"))

            ###### Create the network

            net = MLP()
            net.init([patch_size**2, patch_size**2, patch_size**2, patch_size**2, 135])

            print net

            # image = PIL.Image.fromarray(tile_raster_images(X=net.get_layer(0).get_layer_block(0).w.get_value().reshape((10,-1)),
            #                                                img_shape=(5, 5), tile_shape=(3, 3),
            #                                                tile_spacing=(1, 1)))
            # image.save(self.path + "filters_corruption_30.png")

            ###### Configure the trainer

            # Cost function
            cost_function = CostNegLL()

            # Learning update
            learning_rate = 0.05
            momentum = 0.5
            lr_update = LearningUpdateGDMomentum(learning_rate, momentum)

            # Create monitors and add them to the trainer
            err_validation = MonitorErrorRate(1, "Validation", ds_validation)
            dice_validation = MonitorDiceCoefficient(1, "Validation", ds_validation, 135)

            # Create stopping criteria and add them to the trainer
            max_epoch = MaxEpoch(300)
            early_stopping = EarlyStopping(err_validation, 5, 0.99, 5)

            # Create the network selector
            params_selector = ParamSelectorBestMonitoredValue(err_validation)

            # Create the trainer object
            batch_size = 200
            t = Trainer(net, cost_function, params_selector, [max_epoch, early_stopping],
                        lr_update, ds_training, batch_size,
                        [err_validation, dice_validation])


            ###### Train the network

            t.train()

            ###### Plot the records

            error_rates[idx] = err_validation.get_minimum()
            dice_coeffs[idx] = dice_validation.get_maximum()

        print error_rates
        print dice_coeffs

        plt.figure()
        plt.plot(range_patch_size, error_rates, label="Validation error rates")
        plt.plot(range_patch_size, dice_coeffs, label="Validation dice coefficient")

        plt.xlabel('Patch size')
        plt.savefig(self.path + "res.png")
        tikz_save(self.path + "res.tikz", figureheight = '\\figureheighttik', figurewidth = '\\figurewidthtik')


if __name__ == '__main__':

    exp = ExperimentBrain("final_exp_size_patch")
    exp.run()