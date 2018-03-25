# Rare-Event Detection using Machine Vision and Deep Learning
**Team Members**: Roberta Beaulieu, Rakeem Durand, Thomas Fuller, Ihsan Hashem, and Karim Rattani

## Overview
Surveillance cameras have become a part of our life. We see them in almost every corner of the street and outside restaurants. However, hiring people to monitor all the surveillance video is expensive. The purpose of our project is to implement an algorithm that will detect abnormal activity such as robbery, accident, etc. and alert the corresponding authorities.

## Milestones
 - [x] Perform academic literature review
 - [x] Define the scope of the project
 - [x] Design class structure and interfaces
 - [x] Select algorithm/model to implement
 - [x] Select the dataset we will work with
 - [ ] Implement the algorithm/model chosen
 - [ ] Prepare the dataset
 - [ ] Perform training/testing on model
 - [ ] Hyper-parameterize/optimize the model
 - [ ] Final evaluation and analysis of the model

## Installation and Setup
Our project is written mostly in Python, taking advantage of several data science and machine learning libraries such as Numpy, Pandas, Tensorflow, etc.

### Install dependencies using Anaconda
Included in the repo is a conda environment file called `environment.yml`. This can be used to install all dependencies for the project. First, install Anaconda or Miniconda using this [guide](https://conda.io/docs/user-guide/install/index.html). **Make sure to set your environment vars to use the `conda` command.** Then, simply navigate to the folder and run `conda env create -f environment.yml`. All dependencies should now be installed.

### Download the C3D Pre-trained Model
In order to run the application, you will need to download the pre-trained C3D model (~300 MB) and place it in the `/pretrained_models` folder. The pre-trained model is hosted [here](https://www.dropbox.com/s/u5fxqzks2pkaolx/c3d_ucf101_finetune_whole_iter_20000_TF.model?dl=0).

## Using the Application
**TODO: Include a detailed description on how to use the system.**

## Testing the Application
This project uses the `unittest` module included in Python's core library. All tests are stored in the `/test` directory. Each test class **must** have `_test` appended to the end of the `.py` file (i.e. `mytestclass.py` doesn't work, while `mytestclass_test.py` does). To run the tests you can run the command `python -m unittest discover -v -s test -p *_test.py` in the working directory.
