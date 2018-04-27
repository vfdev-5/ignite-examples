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
 
## Simple blending

Dataset is splitted already on two splits: train and validation. 
Complete procedure is the following:

- Fit and predict with networks:
    - Fit network 1 on the train dataset
    - Predict probabilites with the network 1 on the validation and test datasets
    - Fit network 2 on the train dataset
    - Predict probabilites with the network 2 on the validation and test dataset
    
    ...
    
    - Fit network `n` on the train dataset
    - Predict probabilites with the network `n` on the validation and test dataset

- Create probabilites dataset composed of predictions on the validation dataset
    - Concatenate predictions of single networks 

- Create probabilites dataset composed of predictions on the test dataset
    - Concatenate predictions of single networks 

- Fit a meta-model on the validation probabilites dataset 
    - Use cross-validation to estimate the performance and tune hyperparameters        
    - Finally, train on the whole dataset 

- Predict classes with trained meta-model on the test probabilites dataset

