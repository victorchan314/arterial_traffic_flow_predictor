# Arterial Traffic Flow Prediction

### Random notes about DCRNN

- Data loading is weird. At first glance, it seems to only run through the data once, replicating the last value if necessary, and if it runs out of data, does not supply any more data. This might be avoided if multiple DataLoaders are created.
- Might want to bootstrap (at least the replications)
- StandardScaler only scales the first dimension of the input (maybe to leave time unaffected). Might want to change it so that time is the first dimension and everything else is scaled, or time is the second dimension and everything else is scaled.
