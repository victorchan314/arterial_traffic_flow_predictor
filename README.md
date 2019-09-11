# Arterial Traffic Flow Prediction

This repository contains all of the code that I have been using for my research.

Code to run the training data generation script

`python3 scripts/generate_training_data.py --intersection 5083 --plan_name P2 --output_dir test --timestamps_dir test -v`

### Random notes about DCRNN

- Data loading is weird. At first glance, it seems to only run through the data once, replicating the last value if necessary, and if it runs out of data, does not supply any more data. This might be avoided if multiple DataLoaders are created.
- Might want to bootstrap (at least the replications)
- StandardScaler only scales the first dimension of the input (maybe to leave time unaffected). Might want to change it so that time is the first dimension and everything else is scaled, or time is the second dimension and everything else is scaled.
- I think the way the word "epoch" is used is incorrect; not a huge deal, but just misleading
- dcrnn_cell:103: Could be (1 - u) * state + u * c; in theory is the same, though
- dcrnn_cell:165: Is there an extra 2 times the last extraneous term in the sum?
- dcrnn_model:49: It looks like the decoding cell is the same cell as the encoding cell, which may not perform as well, according to Sutskever's paper
- dcrnn_model:39: Why are the labels using input_dim instead of output_dim?
- Validation loss is used for early stopping regularization
- Change DCRNN so that it predicts with the shape (num_data, offsets, num_detectors, num_dimensions) and that time is not included in the output

### TODO

- Baseline methods visualization and table
- Predictions on dummy data that is linear
- Try predictions for only 1 or 2 detectors with full coverage
- Flow and occupancy and flow / occupancy
- Visualization library
