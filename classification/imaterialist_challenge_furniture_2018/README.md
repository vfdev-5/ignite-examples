# iMaterialist Challenge Furniture 2018 - Playground 

Example of model training/validation/predictions on [iMaterialist Challenge Furniture 2018](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018) 
dataset.

**/!\ Python3 only code**
 
## Download data from Kaggle

```bash
kaggle competitions download -c imaterialist-challenge-furniture-2018
```
and download images to your local storage:
```bash
python utils/download_images.py train.json input/train
python utils/download_images.py validation.json input/validation
python utils/download_images.py test.json input/test
```
Total dataset size is ~ 110GB

### Resize datasets
```bash
python utils/resize_dataset.py input/train 224 input/train_224x224
```

## Training a single model

Edit a configuration file, for example `train_vgg16_bn_basic.py` and simply run
```bash
python train.py configs/your_config.py
```

## Predictions with a single model

Edit a configuration file, for example `predict_vgg16_bn_basic.py` and simply run
```bash
python predict.py configs/your_config.py
```

 




