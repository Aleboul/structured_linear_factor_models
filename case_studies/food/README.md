# Dietary intakes data

## Overview
We evaluate our method on the NHANES 2015–2016 Day 1 total nutrient intake dataset (DR1TOT I; 𝑁 = 9544), publicly available at https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOTI.xpt.

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

#### `dam.py`
Apply DAMEX algorithm to the dataset to find extremal directions.

#### `clef.py`
Apply CLEF algorithm to the dataset to find extremal directions.

### Data Directory (`data/`)

**Data Description:**
- `data.py`: Charge the data available at https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOTI.xpt that contains dietary recall data from the National Health and Nutrition Examination Survey (NHANES) and select the selected six nutrients and give back `nhanes_dr1tot.csv`.

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