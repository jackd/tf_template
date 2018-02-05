# tf_template
Template project for building/evluating/visualizing tensorflow models.

# Getting started
```
cd /path/to/parent_dir
git clone https://github.com/jackd/tf_template.git
```
To run, ensure the directory in which this repository is cloned is on your `PYTHON_PATH`.
```
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```

## Scripts
All files in `scipts` directory are intended to be run. All require a `MODEL_ID` argument first, e.g.
```
python train.py example
```
will train the model with ID `'example'` for the default number of steps.

## `ModelBuilder`s and Serialization
The abstract base class `ModelBuilder` (from `model/builder.py`) defines a minimal interface for building models in a modular fashion, and includes boilerplate code that ties everything together. It is basically an umbrella class combining data processing and a `tf.estimator.Estimator`.

Each implementation of `ModelBuilder` should be able to build data tensors and estimator specifications from JSON serializable meta parameters (nested `dict`s/`list`s of `int`s, `floats`, `str`ings, `None` (as null) and `bool`s), e.g. learning rates, batch sizes, number of nodes in layers, number of layers etc. These meta-parameters should be saved in a `model/params/MODEL_ID.json` file.

To avoid heavy branching of code inside each function of a `ModelBuilder` implementation, significantly different `ModelBuilder`s should be implemented seperately. We call these seperate implementations families.

To aid serialization, we include the `get_builder` function, which loads parameters and calls the correct implementation constructor based on the top-level `'family'` meta parameter.

## Custom `ModelBuilder`s
To implement your own `ModelBuilder`, simply create a class that extends `ModelBuilder` and implement the necessary abstract methods. To make this builder available from `get_builder`, simply call `register_builder_family`. To ensure all scripts call the appropriate registration functions, modify the `setup.register_families` method to register any custom families you have implemented.

Once you have run the examples, you can also remove the example family registration from `setup.register_families`. This will save a small amount of time each script call since code in `model/example_builder` won't have to be run.

## Profiling and simple default testing
For testing and profiling, (`scripts/test_model.py` and `scripts/profile.py` respectively) clone the [tf_toolbox repository](https://github.com/jackd/tf_toolbox) into a directory on your `PYTHON_PATH`.

# Example implementation
See `model/example_builder.py` for an example implementation.

# Tensorboard
To view training summaries, run
```
tensorboard --logdir=model/estimator/MODEL_ID
```
