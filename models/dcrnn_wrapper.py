import os
import sys
import yaml

parent_dir = os.path.abspath("/Users/victorchan/Desktop/UC Berkeley/Research/Code")
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf

from models.model import Model
from lib import utils

from DCRNN.lib.utils import load_graph_data
from DCRNN.model.dcrnn_supervisor import DCRNNSupervisor

class DCRNN(Model):
    """Wrapper that runs DCRNN and saves the results"""
    def __init__(self, *args, config_filename=None, output_filename=None, use_cpu_only=False, **kwargs):
        if not config_filename:
            raise ValueError("config_filename is a required argument")

        with open(config_filename) as f:
            supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        self.tf_config = tf.ConfigProto()
        if use_cpu_only:
            self.tf_config = tf.ConfigProto(device_count={'GPU': 0})
        self.tf_config.gpu_options.allow_growth = True
        self.supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        super(DCRNN, self).__init__(*args, **kwargs)
        self.command = self.base_command.format(config_filename)
        if output_filename:
            self.command += self.output_file_append.format(output_filename)

        print(self.command)

    def _train(self):
        with tf.Session(config=self.tf_config) as sess:
            self.supervisor.train(sess=sess)

    def predict(self, x):
        raise NotImplementedError
