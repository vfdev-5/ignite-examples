# Tiny ImageNet 200 Playground

Train and evaluate models with [ignite](https://github.com/pytorch/ignite)

More detail on the dataset can be found [here](https://tiny-imagenet.herokuapp.com/). 
Dataset [download link](http://cs231n.stanford.edu/tiny-imagenet-200.zip) 

## Requirements

We need PyTorch (can be installed from http://pytorch.org/) and the following dependencies:

```bash
pip install --upgrade -r requirements.txt
```

## Training

Checkout training configuration, for example, `configs/train_vgg16_bn_basic.py` and update paths.
Next, start training with a simple command:
```bash
python tiny_imagenet200_train_playground.py configs/train_vgg16_bn_basic.py
```

The output folder will contain folders of with training runs:
- `training_YYYYmmDD_HHMM`
    - `train.log` : training log
    - 5 best models, `model_*.pth`
    - 1 last model, `checkpoint_*.pth`
    - tensorboard logs

### TensorboardX training monitoring

```bash
tensorboard --logdir=tiny_imagenet200_output
``` 

## Evaluation

Same as in the training part, edit a conifguration file, for example, `configs/test_vgg16_bn_basic.py`
and run the following script:
```bash
python tiny_imagenet200_test_playground.py configs/test_vgg16_bn_basic.py
``` 

Predictions are stored in a CSV file in the output folder.


 