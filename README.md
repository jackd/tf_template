# tf_template
Template project for building/evluating/visualizing tensorflow models. Mostly just a structured wrapper around `tf.estimator.Estimator` with additional functionality for visualizing inputs/predictions, simple tests and profiling.

## Setup
```
cd /path/to/parent_dir
git clone https://github.com/jackd/tf_template.git
```
To run, ensure the directory in which this repository is cloned is on your `PYTHON_PATH`.
```
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```

Creating profiles and running tests also requires the [tf_toolbox repository](https://github.com/jackd/tf_toolbox)
```
git clone https://github.com/jackd/tf_toolbox.git
```

## Usage
We define 4 helper classes to encourage modularity:
* [DataSource](./data_source.py): for creating data pipelines and visualizing, independently of trained models.
* [InferenceModel]('./inference_model.py'): for creating trainable component of a model, e.g. `features -> logits` for classification.
* [EvalModel]('./eval_model.py'): for evaluating a trained model, and calculating the inference loss.
* [TrainModel]('./train_model.py'): for specifying the optimization strategy, `batch_size` and `max_steps`.

As the name suggests, the [Coordinator]('./coordinator.py') is for coordinating all of the above. [cli.py](./cli.py) provides some command line interface helpers using `absl.flags`.

## Example
See the [MNIST example](./example/mnist) for a full working example.
```
cd tf_template/example/mnist/script
./main.py --action=vis_inputs  # or ./vis_inputs.py
./main.py --action=test
./main.py --action=profile
./main.py --action=train
./main.py --action=vis_predictions
./main.py --action=evlauate
tensorboard --logdir=../_models
```

To use the `big` network (specified in `tf_template/example/mnist/params/big.json`) just specify `--model_id=big`
```
./main.py --model_id=big --action=train
```

To run your own model, just create a new `.json` file in `tf_template/example/mnist/params/`. See `tf_template/example/mnist/coordinator.py` for deserialization.


## Starting your own project
1. Copy the `new_project` subdirectory and rename all references to `new_project` to `my_fancy_project` and `NewProject` to `MyFancyProject`.
```
cd tf_template
cp -r project ../my_fancy_project
sed -i 's/NewProject/MyFancyProject/g' *
sed -i 's/new_project/my_fancy_project/g' *
```
2. Find all `TODO` strings and get to work!
