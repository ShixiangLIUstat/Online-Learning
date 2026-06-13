# High-dimensional-online-learning-via-asynchronous-decomposition

This repository contains the R code for reproducing the simulation studies and real-data analyses presented in the manuscript "High-dimensional online learning via asynchronous decomposition", available at https://arxiv.org/abs/2603.20696

## File Description

- **funs.R**
  Implementation of the online learning methods considered in the paper, including AD-IHT, AD-Lasso, Renew-Lasso, Renew-SIM, and RADAR-GLM.

- **logistic simulation.R**  
  Simulation code for high-dimensional online logistic regression, including two online settings:
  1. fixed sample size for each batch;
  2. exponentially increasing sample size across batches.

- **logistic plot.R**  
  Visualization code for the logistic regression simulation results.

- **poisson simulation.R**  
  Simulation code for high-dimensional online Poisson regression under two online settings:
  1. fixed sample size for each batch;
  2. exponentially increasing sample size across batches.
 
  The corresponding results are reported in the Supplementary Material.

- **poisson plot.R**  
  Visualization code for the Poisson regression simulation results reported in the Supplementary Material.

- **FinBase.R**  
  Implementation of the online learning methods adapted for financial data analysis.

- **Financial analysis.R**  
  Main script for the financial data analysis, including data preprocessing, parallel computation, and result aggregation.

## Requirements

- R >= 4.0
- Required packages: glmnet, snowfall, dplyr, parallel, mvnfast, mccr, caret, mltools, Matrix, stringr, ggplot2, tidyr
