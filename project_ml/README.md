Homework 01
==============================

A short description of the project.

Project Organization
------------
    ├── configs
    │   └── feature_params     <- Configs for features
    │   └── path_config        <- Configs for all needed paths
    │   └── splitting_params   <- Configs for splitting params
    │   └── train_params       <- Configs for logreg and randomforest models parametres
    │   └── predict_config.yaml   <- Config for prediction pipline
    │   └── train_config.yaml   <- Config for train pipline
    │ 
    ├── data               <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts for splitting dataset to train and test
    │   │   └── make_dataset.py
    │   │
    │   ├── entities       <- Scripts for creating dataclasses
    │   │    
    │   │
    │   ├── features              <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    |   |   └── custom_scaler.py  <- Custom scaler transformer
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   |
    │   ├── outputs       <- Hydra logs
    │   │   
    │   ├──  utils        <- Scripts for serialized models, reading data
    │   |    └── utils.py
    |   |
    |   ├── predict_pipeline.py   <- pipeline for making predictions
    |   |
    |   └── train_pipeline.py     <- pipeline for model training
    |
    ├── tests              <- tests for the project
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    ├── LICENSE
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── README.md          <- The top-level README for developers using this project.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------
Train model
------------
Model training process relies on configs:

    ├── configs
    │   └── feature_params
    │   │   └── features.yaml   <- features name and feature categories and normalize parameter for 
    |   |                            numerical features.
    │   |
    │   ├── path_config           
    │   │   └── path_config.yaml <- Path for data, models, artefacts
    │   │
    │   ├── splitting_params
    │   │   └── splitting_params.yaml <- Config with train/val split of data
    │   │
    │   ├── train_params
    |   |   └── rf.yaml          <- RandomForestClassifier model parameters
    │   │                     │   │
    │   ├── train_config.yaml      <- Config for train pipeline (for Hydra)

Training command:  `python src/train_pipeline.py`

--------
Prediction
--------
Configs for predictions

    ├── configs
        └── predict_config.yaml <- Config with paths to models and artefacts
        
Make prediction with command:  `python src/predict_pipeline.py`