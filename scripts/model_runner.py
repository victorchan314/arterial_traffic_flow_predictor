import argparse
import copy
from multiprocessing import Process
import os
from string import Template
import sys

parent_dir = os.path.abspath(".")
sys.path.append(parent_dir)

import numpy as np

from models import models
from lib import data_utils
from lib import utils

DATA_CATEGORIES = ["train", "val", "test"]

def load_data(data_directory, verbose=0):
    data = {}
    if verbose:
        print("Loading data from directory {}".format(data_directory))

    for category in DATA_CATEGORIES:
        path = "{}/{}.npz".format(data_directory, category)
        if verbose > 1:
            print("Loading datafile at path {}".format(path))

        with open(path, "rb") as f:
            npz = np.load(f)
            data["{}_{}".format(category, "x")] = npz["x"]
            data["{}_{}".format(category, "y")] = npz["y"]

            if verbose > 2:
                print("{}_{} shape: {}".format(category, "x", npz["x"].shape))
                print("{}_{} shape: {}".format(category, "y", npz["y"].shape))

    return data

def get_augment_function(data_augment_dict):
    functions = []
    if "zero" in data_augment_dict:
        zero = data_augment_dict["zero"]

        if "detectors" in zero:
            from scripts.generate_training_data import get_sensors_list
            detectors = zero["detectors"]
            detector_list = get_sensors_list(detectors["detector_list"])
            zero_detector_function = data_utils.zero_out_detectors(detectors["detectors"], detector_list)
            functions.append(zero_detector_function)
        if "days" in zero:
            proportion = zero["days"]["proportion"]
            seed = zero["days"].get("seed", 1)
            zero_days_function = data_utils.zero_out_days(proportion, seed)
            functions.append(zero_days_function)

    augment_function = utils.concatenate_functions(functions)

    return augment_function

def load_models(model_names):
    return [getattr(models, model_name) for model_name in model_names]

def run_models(data, model_configs, model_order=None, overwrite=False, debug=False, verbose=0):
    train_x = data["train_x"]
    train_y = data["train_y"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    model_names = list(model_configs.keys()) if model_order is None else model_order
    models = load_models(model_names)

    if verbose:
        print("Starting to run models...")

    if verbose > 2:
        print("Max train_y: {}".format(np.max(train_y)))
        print("Min train_y: {}".format(np.min(train_y)))
        print("Max val_y: {}".format(np.max(val_y)))
        print("Min val_y: {}".format(np.min(val_y)))
        print("Max test_y: {}".format(np.max(test_y)))
        print("Min test_y: {}".format(np.min(test_y)))

    for i in range(len(models)):
        model_name = model_names[i]
        model_class = models[i]

        if isinstance(model_configs[model_name], list):
            model_kwargs = model_configs[model_name]
        elif isinstance(model_configs[model_name], dict) and model_configs[model_name].get("__named_models__", False):
            model_kwargs = [v for k, v in model_configs[model_name].items() if k != "__named_models__"]
        else:
            model_kwargs = [model_configs[model_name]]

        for kwargs in model_kwargs:
            if not overwrite and os.path.exists(kwargs["base_dir"])\
                    and "predictions.npz" in os.listdir(kwargs["base_dir"]):
                if verbose > 1:
                    print("Predictions exist in {}; skipping this model to not overwrite".format(kwargs["base_dir"]))

                continue
            elif debug:
                print("Debug: running model at base_dir {}".format(kwargs["base_dir"]))
                continue

            model = model_class(train_x, train_y, val_x, val_y, test_x, test_y,
                                verbose=verbose, **kwargs)
            if verbose:
                print("Created {} model".format(model_name))

            try:
                model.train()
            except Exception as e:
                model.close()
                print(e)
                continue

            if verbose:
                print("Trained {} model".format(model_name))

            errors = model.get_errors()

            if verbose > 1:
                for category in DATA_CATEGORIES:
                    if category in errors:
                        for key, value in errors[category].items():
                            print("{} {}: {}".format(category, key, value))

            if "base_dir" in kwargs:
                base_dir = kwargs["base_dir"]
                utils.verify_or_create_path(base_dir)
                path = os.path.join(base_dir, "predictions.npz")
                model.save_predictions(path)

            model.close()

def run_config(config, debug=False, verbose=0):
    data_directory = config["data_directory"]
    data = load_data(data_directory, verbose=verbose)
    model_configs = config["models"]
    model_order = config.get("model_order")
    data_augment = config.get("data_augment", {})
    overwrite = config.get("overwrite", False)

    if model_configs:
        if data_augment:
            augment_function = get_augment_function(data_augment)
            data_augmenter = utils.DataAugmenter(data_directory, augment_function)
            data_augmenter.copy()

            try:
                run_models(data, model_configs, model_order=model_order, overwrite=overwrite, debug=debug, verbose=verbose)
            finally:
                data_augmenter.restore()
        else:
            run_models(data, model_configs, model_order=model_order, overwrite=overwrite, debug=debug, verbose=verbose)

def run_configs(configs, debug=False, verbose=0):
    for config in configs:
        run_config(config, debug=debug, verbose=verbose)

def loop_config(config, debug=False, verbose=0):
    loop = config["loop"]
    values = loop["values"]
    keys = loop["keys"]
    is_parallel = "parallel" in loop
    exclude = loop.get("exclude", [])

    loop_mappings = utils.dictionary_product(values)

    if "substitute" in loop:
        for k, v in loop["substitute"].items():
            for mapping in loop_mappings:
                mapping[k] = v["map"][mapping[v["key"]]]

    if is_parallel:
        parallel = loop["parallel"]
        parallel_values = {value: values[value] for value in parallel}
        parallel_keys = utils.ordered_tuple_product(parallel_values, parallel)
        configs = {key: [] for key in parallel_keys}

    for mapping in loop_mappings:
        itr_config = copy.deepcopy(config)
        for loop_key in keys:
            if isinstance(loop_key, str):
                key = loop_key
                type_cast = str
            else:
                key = loop_key[0]
                type_cast = utils.get_type_by_name(loop_key[1])

            c = itr_config
            config_path = key.split("/")
            for k in config_path[:-1]:
                c = c[k]

            template = Template(c[config_path[-1]])
            c[config_path[-1]] = type_cast(template.substitute(mapping))

        for exclusions in exclude:
            exclude_values = exclusions["values"]
            if all([exclude_values[k] == mapping[k] for k in exclude_values]):
                c = itr_config
                key = exclusions["key"]
                config_path = key.split("/")
                for k in config_path[:-1]:
                    c = c[k]

                del c[config_path[-1]]

        if is_parallel:
            mapping_key = tuple(mapping[value] for value in parallel_values)
            configs[mapping_key].append(itr_config)
        else:
            run_config(itr_config, verbose=verbose)

    if is_parallel:
        processes = []
        for itr_configs in configs.values():
            p = Process(target=run_configs, args=(itr_configs,), kwargs={"debug": debug, "verbose": verbose})
            processes.append(p)

        utils.run_process_list_parallel(processes)



def main(args):
    verbose = args.verbose
    config = utils.load_yaml(args.config)
    debug = args.debug

    if "loop" in config.keys():
        loop_config(config, debug=debug, verbose=verbose)
    else:
        run_config(config, debug=debug, verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file to specify data and model")
    parser.add_argument("-d", "--debug", action="store_true", help="Execute model runner without running models")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")
    args = parser.parse_args()

    main(args)
