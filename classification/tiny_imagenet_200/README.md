# Tiny ImageNet 200 Playground

Train and evaluate models with [ignite](https://github.com/pytorch/ignite)

More detail on the dataset can be found [here](https://tiny-imagenet.herokuapp.com/). 
Dataset [download link](http://cs231n.stanford.edu/tiny-imagenet-200.zip) 

## Requirements

We need PyTorch (can be installed from http://pytorch.org/) and the following dependencies:

```bash
pip install --upgrade git+https://github.com/pytorch/vision.git
pip install --upgrade git+https://github.com/pytorch/ignite.git
pip install --upgrade git+https://github.com/lanpa/tensorboard-pytorch.git
pip install --upgrade numpy scikit-learn
```

if you want to use NASNet-A Mobile, consider to install
```bash
pip install --upgrade git+https://github.com/Cadene/pretrained-models.pytorch.git
```  

## Training

Start training with a simple command:
```bash
python tiny_imagenet200_train_playground.py --output=tiny_imagenet200_output
```
for more options:
```bash
ptyhon tiny_imagenet200_train_playground.py --help
```

The output folder will contain folders of with training runs:
- `training_YYYYmmDD_HHMM`
    - `train.log` : training log
    - 5 best models, `model_*.pth`
    - tensorboard logs

### TensorboardX training monitoring

```bash
tensorboard --logdir=tiny_imagenet200_output
``` 

## Evaluation


 