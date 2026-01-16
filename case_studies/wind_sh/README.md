# Wind gusts in Schleswig-Holstein

## Overview
We aim at analyzing tail dependence between hourly peak wind speeds (m/s) from $d = 22$ weather stations in Schleswig-Holstein (northern Germany) during winter seasons (December-January-February) from December 2013 through February 2018.
## Project Structure

### Core Python Files

#### `main.py`
The main execution script that orchestrates the entire analysis pipeline. This file:
- Loads and preprocesses data
- Computes extremal correlation matrices
- Calibrates model parameters (kappa values)
- Estimates factor loading matrices
- Generates visualizations and saves results

#### `est_impure.py`
Contains functions for estimating the factor loading matrix A.

#### `est_pure.py`
Dedicated functions for estimating pure variables.

### Data Directory (`data/`)

**Data Description:**
- `hourly_ff_schleswig_holstein_matrix.csv`: Contains hourly wind speed from 22 weather stations in Schleswig-Holstein.
- `stations.csv`: contains all the stations id, Temporal coverage, Spatial coverage, Stations name and Bundesland.
- `data.py`: Download the data for the selected weather stations from the website: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/extreme_wind/historical/ and give back `hourly_ff_schleswig_holstein_matrix.csv`.

### Results Directory (`results/`)

## Required Packages

numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
networkx>=2.5.0
tqdm>=4.60.0

## Usage

### Basic Execution

python main.py