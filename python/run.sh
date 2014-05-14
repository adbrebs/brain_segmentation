#!/bin/bash

python main_data_generation.py creation_training_data_1.ini
python main_data_generation.py creation_testing_data_1.ini
python main_train.py training.ini
