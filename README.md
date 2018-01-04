# Dog-Breed-Identification

A [competition](https://www.kaggle.com/c/dog-breed-identification) from Kaggle. Determine the breed of a dog in an image.

## Create your own model

create the model in networks folder and tune the parameters in config/param.yml file

## Structure

```bash
root
├── README.md
├── config
│   └── param.yml
├── main.py
├── cnn_main.py 
├── input (all the files are downloaded from Kaggle)
│   ├── labels.csv
│   ├── sample_submission.csv
│   ├── train
│   └── test
├── dataset
│   └── dataset.h5
├── networks
│   ├── __init__.py
│   └── model.py
└── utils
    ├── DataPreprocessing.py
    └── __init__.py
```
