
# TFM-RUL-CMAPSS

# Remaining Useful Life Prediction of Turbofan Engines

This repository contains the **R scripts and main results** developed as part of a **Master's Thesis in Industrial Organization Engineering** at Universidad Internacional de Valencia (VIU).

## Project Overview

The project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan engines using machine learning techniques and NASA's **C-MAPSS FD002 dataset**.

Three predictive models were developed and evaluated:

- **Random Forest**
- **XGBoost**
- **Multilayer Perceptron (MLP)**

The methodology includes:

- Data preprocessing
- Exploratory data analysis
- Hyperparameter tuning
- Model training
- Performance evaluation
- Error analysis

## Repository Structure

- **`scripts/`**: R scripts used throughout the study.
- **`resultados/tablas/`**: Tables and numerical results generated during the analysis.
- **`resultados/figuras/`**: Figures and visualizations generated during the analysis.

## Scripts Description

The R scripts are organized according to the different stages of the methodology developed in the Master's Thesis.

| Script | Description |
|---|---|
| `01_preparacion_datos.R` | Data quality checks and construction and verification of the RUL variable. |
| `02_analisis_exploratorio.R` | Exploratory data analysis, descriptive statistics, correlation analysis and data visualization. |
| `03_preprocesamiento_modelado.R` | Analysis of operating conditions and identification of relevant predictor variables. |
| `04_preparacion_datos_modelado.R` | Preparation of training and testing data for predictive modeling. |
| `05a_ajuste_hiperparametros.R` | Hyperparameter tuning using five-fold cross-validation grouped by engine. |
| `05b_entrenamiento_RF.R` | Training of the Random Forest model using the selected hyperparameters. |
| `05c_entrenamiento_XGBoost.R` | Training of the XGBoost model using the selected hyperparameters. |
| `05d_entrenamiento_MLP.R` | Training of the Multilayer Perceptron neural network. |
| `06_evaluacion_comparacion_modelos.R` | Evaluation and comparison of the three models using MAE, RMSE and R². |
| `07_analisis_errores_RF.R` | Detailed prediction error analysis of the selected Random Forest model. |

### Additional Scripts

The following scripts document complementary stages of model development:

| Script | Description |
|---|---|
| `05_entrenamiento_modelos.R` | Combined implementation of model training and validation procedures. |
| `05b_ajuste_MLP.R` | Evaluation of different hidden-layer sizes for the MLP neural network. |

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset — FD002**

The study uses the FD002 subset, which contains simulated turbofan engine degradation data under different operating conditions.

The repository includes prepared datasets used in the analysis:

- `train_FD002_prepared.csv`
- `test_FD002_prepared.csv`
- `RUL_FD002_prepared.csv`

The original NASA dataset is not included in this repository.

## Software

The analysis was carried out using **R and RStudio**.

The main R packages used include:

- `tidyverse`
- `ranger`
- `xgboost`
- `nnet`
- `ggplot2`
- `moments`

## Execution Instructions

The scripts are intended to be executed in RStudio, following the order established by the methodology.

1. Download or clone the repository.
2. Open the project in RStudio and set the repository root as the working directory.
3. Install the required R packages.
4. Execute the data preparation and exploratory analysis scripts.
5. Prepare the modeling datasets and perform hyperparameter tuning.
6. Train the selected Random Forest, XGBoost and MLP models.
7. Execute the model evaluation and error analysis scripts.

Some scripts require intermediate files generated during previous stages, including trained model files in `.rds` format.

**Note:** The scripts are currently being adapted to use relative file paths. Full reproducibility from a clean download has not yet been verified.

## Academic Purpose

This repository provides supporting material for the Master's Thesis and facilitates the **consultation and reproducibility** of the methodology and results.

The scripts and results are made available for academic and research purposes.
