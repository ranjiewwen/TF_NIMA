
# NIMA: Neural Image Assessment base on tensorflow

## Get started

- first you need download the vgg pre-train model weight from :[vgg16_weights.npz](http://www.cs.toronto.edu/~frossard/post/vgg16/)
- then put it into the directory: data/vgg_models/
- then you shold prepare the datasets list throgh data/* scripts
- then you need modity the BASE_PATH in the code which is your iqa dataset root path
- final you can run the code, make sure you have follow env : ubuntu16.04+tensorflow1.8+cuda9.0
- by the way, you can modify the parameters in `tools/train_nima.py` funtions `process_command_args`

## Run the code

- first train the models: run the `tools/train_nima.py` scripts.
- second test the models: run the `tools/evaluate.py` scripts.
- last predict images: run the `demo/predict.py` scripts.

## Experiments result

- you can see train/test log from directory `experiments/datasets/`
- tensorboard logs in `experiments/datasets/logs`
- save train modes in `experiments/datasets/experiment_name`
