import os
import sys
import yaml

parent_dir = os.path.abspath("/Users/victorchan/Desktop/UC Berkeley/Research/Code")
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf

from models.model import Model
from lib import data_utils
from lib.utils import Tee

from DCRNN.lib.utils import load_graph_data
from DCRNN.model.dcrnn_supervisor import DCRNNSupervisor

class DCRNN(Model):
    """Wrapper that runs DCRNN and saves the results"""
    def __init__(self, *args, config_filename=None, output_filename=None, predictions_filename=None,
                 use_cpu_only=False, **kwargs):
        if not config_filename:
            raise ValueError("config_filename is a required argument")

        self.config_filename = config_filename
        self.output_filename = output_filename
        self.predictions_filename = predictions_filename

        with open(config_filename) as f:
            self.supervisor_config = yaml.load(f)

        graph_pkl_filename = self.supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, self.adj_mx = load_graph_data(graph_pkl_filename)

        self.tf_config = tf.ConfigProto()
        if use_cpu_only:
            self.tf_config = tf.ConfigProto(device_count={'GPU': 0})
        self.tf_config.gpu_options.allow_growth = True
        self.supervisor = DCRNNSupervisor(adj_mx=self.adj_mx, **self.supervisor_config)

        super(DCRNN, self).__init__(*args, **kwargs)

    def _train(self):
        if self.output_filename:
            tee = Tee()

        with tf.Session(config=self.tf_config) as sess:
            self.supervisor.train(sess=sess)

        if self.output_filename:
            del tee

    def predict(self, x):
        with tf.Session(config=self.tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=self.adj_mx, **self.supervisor_config)
            supervisor.load(sess, self.supervisor_config['train']['model_filename'])
            outputs = supervisor.evaluate(sess)
            np.savez_compressed(self.output_filename, **outputs)
            print('Predictions saved as {}.'.format(self.output_filename))
