__author__ = 'adeb'

from spynet.utils.utilities import load_config
from data_brain_parcellation import DataBaseBrainParcellation
from spynet.models.network import Network1
from spynet.training.trainer import Trainer


if __name__ == '__main__':

    ### Load the config file
    training_cf = load_config("cfg_general")

    ### Create the database
    db = DataBaseBrainParcellation()
    db.init(training_cf.prop_validation, training_cf)

    ### Create the network
    net = Network1()
    net.init(n_in=29**2, n_out=139)
    print net

    ### Train the network
    t = Trainer(training_cf, net, db)
    t.train()

    ### Save the network
    net.save_parameters(training_cf.net_path)