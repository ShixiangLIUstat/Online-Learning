# High-dimensional online learning via asynchronous decomposition

This repository contains the R code for reproducing the simulation studies and real-data analyses presented in the manuscript "High-dimensional online learning via asynchronous decomposition", available at https://arxiv.org/abs/2603.20696

## File Description

- **code/funs.R**  
  Implementation of the online learning methods considered in the paper, including AD-IHT, AD-Lasso, Renew-Lasso, Renew-SIM, and RADAR-GLM.

- **code/logistic simulation.R**  
  Simulation code for high-dimensional online logistic regression, including two online settings:
  1. fixed sample size for each batch;
  2. exponentially increasing sample size across batches.

- **code/logistic plot.R**  
  Visualization code for the logistic regression simulation results.

- **code/poisson simulation.R**  
  Simulation code for high-dimensional online Poisson regression under two online settings:
  1. fixed sample size for each batch;
  2. exponentially increasing sample size across batches.
 
  The corresponding results are reported in the Supplementary Material.

- **code/poisson plot.R**  
  Visualization code for the Poisson regression simulation results reported in the Supplementary Material.

- **code/FinBase.R**  
  Implementation of the online learning methods adapted for financial data analysis.

- **code/Financial analysis.R**  
  Main script for the financial data analysis, including data preprocessing, parallel computation, and result aggregation.

- **data/logitfix0424.RData** 
  Simulation results for logistic regression with fixed sample size for each batch.

- **data/logitincrease0418.RData** 
  Simulation results for logistic regression with exponentially increasing sample size across batches. 

- **data/Poissonfix0422.RData** 
  Simulation results for Poisson regression with fixed sample size for each batch.

- **data/Poissonincrease0423.RData** 
  Simulation results for Poisson regression with exponentially increasing sample size across batches. 


Financial Distress data is available at https://www.kaggle.com/datasets/shebrahimi/financial-distress


## Requirements

- R >= 4.0
- Required packages: glmnet, snowfall, dplyr, parallel, mvnfast, mccr, caret, mltools, Matrix, stringr, ggplot2, tidyr
