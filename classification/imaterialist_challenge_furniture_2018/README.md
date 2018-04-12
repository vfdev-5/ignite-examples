# iMaterialist Challenge Furniture 2018 - Playground 

**/!\ Python3 code**
 
## Download data from Kaggle

```bash
kaggle competitions download -c imaterialist-challenge-furniture-2018
```
and download images to your local storage:
```bash
python scripts/download_images.py train.json input/train
python scripts/download_images.py validation.json input/validation
python scripts/download_images.py test.json input/test
```
Total dataset size is ~ 110GB


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

 




