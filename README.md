# Text-Deconfounded Factorization for Selection-Biased Ratings

## Overview
This repository contains the implementation of the **Text-Deconfounded Factorization Model (TDFM)**, a probabilistic method for correcting selection bias in recommender systems. This project was developed for the **Probabilistic Models and Machine Learning** course (STCS 6701) at Columbia University.

The model augments standard Matrix Factorization (MF) with:
1.  **Topic Modeling**: A bag-of-words encoder to capture content preferences from review text.
2.  **Exposure Surrogate**: A Poisson Factorization (PF) proxy to estimate exposure intensity and correct for Missing Not At Random (MNAR) data.

## File Structure
* `TDFM_coding_part_revised_v3.ipynb`: The main notebook containing:
    * Data preprocessing (UCI Online Retail / Recipe dataset subset).
    * Poisson Factorization for exposure surrogate construction.
    * TDFM model implementation (PyTorch).
    * Evaluation loops (RMSE, MAE, and Inverse-Propensity Weighting).

## Key Results
On the aligned test set ($N=2021$ users, $M=100$ items), TDFM outperforms standard Matrix Factorization:

| Model | RMSE | MAE |
| :--- | :--- | :--- |
| Matrix Factorization (Baseline) | 0.6234 | 0.4066 |
| **TDFM (Full)** | **0.5369** | **0.3333** |

## Requirements
* Python 3.8+
* PyTorch
* NumPy, Pandas
* Scikit-learn

## Usage
Run the Jupyter notebook `TDFM_coding_part_revised_v3.ipynb` to reproduce the training pipeline and evaluation metrics.
