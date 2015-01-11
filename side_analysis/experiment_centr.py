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
from data_brain_parcellation import generate_and_save, RegionCentroids
from spynet.data.utils_3d.pick_patch import *


class ExperimentBrain(Experiment):
    """
    Experiment to measure the benefits of the approximated distances to centroids and the point of iterating these
    approximations.
    """
    def __init__(self, exp_name):
        Experiment.__init__(self, exp_name)

    def copy_file_virtual(self):
        copy2(inspect.getfile(inspect.currentframe()), self.path)

    def run(self):
        ###### Create the datasets

        data_path = "./datasets/report_ultimate_random/"

        training_data_path = data_path + "train_2.h5"
        testing_data_path = data_path + "test_2.h5"
        ds_training = DatasetBrainParcellation()
        ds_training.read(training_data_path)
        # image = PIL.Image.fromarray(tile_raster_images(X=ds_training.inputs[0:100],
        #                                                img_shape=(29, 29), tile_shape=(10, 10),
        #                                                tile_spacing=(1, 1)))
        # image.save(self.path + "filters_corruption_30.png")

        prop_validation = 0.5  # Percentage of the testing dataset that is used for validation (early stopping)
        ds_testing = DatasetBrainParcellation()
        ds_testing.read(testing_data_path)
        ds_validation, ds_testing = ds_testing.split_datapoints_proportion(prop_validation)
        # Few stats about the targets
        analyse_classes(np.argmax(ds_training.outputs, axis=1))

        # Scale some part of the data
        s = pickle.load(open(self.path + "s.scaler", "rb"))
        s.scale(ds_testing.inputs)

        ###### Create the network

        # Load the networks
        net1 = NetworkUltimateMLPWithoutCentroids()
        net1.init(29, 13, 135)
        net1.load_parameters(open_h5file(self.path + "net_no_centroids.net"))

        net2 = NetworkUltimateMLP()
        net2.init(29, 13, 134, 135)
        net2.load_parameters(open_h5file(self.path + "net.net"))

        ###### Evaluate on testing

        compute_centroids_estimate(ds_testing, net1, net2, s)


def compute_centroids_estimate(ds, net_wo_centroids, net_centroids, scaler, n_iter=3):
    pred = np.argmax(net_wo_centroids.predict(ds.inputs, 1000), axis=1)
    r = RegionCentroids(134)
    r.update_barycentres(ds.vx, pred)
    p = PickCentroidDistances(134)
    distances = p.pick(ds.vx, None, None, r)[0]
    ds.inputs[:, -134:] = distances
    scaler.scale(ds.inputs)

    # New evaluations
    for i in range(n_iter):
        d = compute_dice(pred, np.argmax(ds.outputs, axis=1), 134)
        print np.mean(d)

        pred = np.argmax(net_centroids.predict(ds.inputs, 1000), axis=1)
        r.update_barycentres(ds.vx, pred)
        distances = p.pick(ds.vx, None, None, r)[0]
        ds.inputs[:, -134:] = distances
        scaler.scale(ds.inputs)


if __name__ == '__main__':

    exp = ExperimentBrain("report_ultimate_random_mlp_true_valid")
    exp.run()