import os
import re

import numpy as np
import tensorflow as tf

from models.model import Model
from lib import data_utils
from lib import utils
from lib.utils import Tee

from DCRNN.lib.utils import load_graph_data
from DCRNN.model.dcrnn_supervisor import DCRNNSupervisor

class DCRNN(Model):
    """Wrapper that runs DCRNN and saves the results"""
    def __init__(self, *args, config_filename=None, is_trained=False, output_filename=None, predictions_filename=None,
                 use_cpu_only=False, base_dir=None, calculate_train_and_val_errors=True, **kwargs):
        super(DCRNN, self).__init__(*args, **kwargs)

        if not base_dir is None:
            base = base_dir.split("/")[-1]
            self.config_filename = os.path.join(base_dir, "{}.yaml".format(base))
            self.predictions_filename = os.path.join(base_dir, "predictions.npz")
        else:
            if not config_filename:
                raise ValueError("config_filename is a required argument")

            self.config_filename = config_filename
            self.predictions_filename = predictions_filename

        self.output_filename = output_filename
        self._is_trained = is_trained
        self.calculate_train_and_val_errors = calculate_train_and_val_errors

        if self.output_filename:
            self.output_filename = os.path.join(base_dir, "{}.out".format(base))
            self.output_filename = output_filename
            self.tee = Tee(self.output_filename)

        self.supervisor_config = utils.load_yaml(self.config_filename)

        graph_pkl_filename = self.supervisor_config['data'].get('graph_pkl_filename', None)
        if graph_pkl_filename is None:
            self.adj_mx = None
        else:
            sensor_ids, sensor_id_to_ind, self.adj_mx = load_graph_data(graph_pkl_filename)

        self.tf_config = tf.ConfigProto()
        if use_cpu_only:
            self.tf_config = tf.ConfigProto(device_count={'GPU': 0})

        self.tf_config.gpu_options.allow_growth = True
        self.supervisor = DCRNNSupervisor(adj_mx=self.adj_mx, **self.supervisor_config)
        self.session = tf.Session(config=self.tf_config)

    def train(self):
        self._train()

    def _train(self):
        if not self._is_trained:
            self._is_trained = True
            self.supervisor.train(sess=self.session)
        else:
            if "train" in self.supervisor_config and "model_filename" in self.supervisor_config["train"]:
                self.supervisor.load(self.session, self.supervisor_config['train']['model_filename'])

        if self.calculate_train_and_val_errors:
            self.get_errors_from_log_file()

        self.predictions = self.predict(None)
        self.errors["test"] = data_utils.get_standard_errors(np.array(self.predictions["groundtruth"]), np.array(self.predictions["predictions"]))

    def predict(self, x):
        # x is not used in this function. It is assumed that x is the test data from the config file.
        return self.supervisor.evaluate(self.session)

    def save_predictions(self, path):
        np.savez_compressed(self.predictions_filename, **self.predictions)
        print('Predictions saved as {}.'.format(self.predictions_filename))

    def get_errors_from_log_file(self):
        if "train" in self.supervisor_config and "log_dir" in self.supervisor_config["train"]:
            path = "{}{}".format(self.supervisor_config["train"]["log_dir"], "info.log")
        else:
            path = self.supervisor._log_dir

        last_errors_message = None
        with open(path) as f:
            for line in f:
                message = line.split(" - ")[-1]
                if message.startswith("Epoch"):
                    last_errors_message = message
                elif line.startswith("Horizon"):
                    pass

        last_errors = re.findall("[A-Za-z_]+: ?[-+]?\d*\.\d+", last_errors_message)
        errors = {}

        for error in last_errors:
            key, value = re.compile(": ?").split(error)
            errors[key] = float(value)

        self.errors["train"] = {"mae": errors["train_mae"]}
        self.errors["val"] = {"mae": errors["val_mae"]}

    def close(self):
        self.session.close()
        tf.reset_default_graph()
        
        if self.output_filename:
            del self.tee
