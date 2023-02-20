# Time Series Event Classification: Detection of formation tops from wellbore data

This project is for the course of Big Data Research Project at CentraleSupélec. This repository contains the data, scripts, and other supporting files used in the project. 

## File Structure

- data : training data of 600 wells
- testdata : test data of 50 wells
- hacktops: a Python package providing some useful utility functions and evaluation methods.
- notebook & notebook1: codes for all exploration and experiments.


## How to run

Create a python3 virtualenv.

    virtualenv -p python3 env

Activate the virtualenv

    source env/bin/activate

Install The Package Hacktop

    pip install -e .

Install other dependency libraries, such as

    pip install jupyter
    pip install plotly
    pip install dtaidistance
    pip install (other missing libraries)

Run Jupyter notebook and execute notebooks

    jupyter notebook

Note: Make sure the virtual environment is activated when you are playing with the notebooks