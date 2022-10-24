# Predicting Cancer Rates from Environmental Justice Screen Data

## Problem Statement
A variety of characteristics can contribute to increased cancer rates, including environmental influences and demographic characteristics. Using the Environmental Justice Screen data provided by the United States Environmental Protection agency, we want to build regression models to predict the air toxics cancer rates per million person in each census block group using other features such as demographic information and environmental features. Additionally, we want to compare the performance of those models to a model built using only environmental feature data. This is informed by prior knowledge that low income areas and communities of color often are located near environmental hazards and have increased vulnerability to negative effects such as greater cancer risk.

We are interested in predicting air toxics cancer rates to help the United States government budget for healthcare costs in different areas of the United States that Medicare and Medicaid will help cover. 

The question we are trying to answer is what the best regression model is to predict air toxics cancer rates per million person given data on environmental and demographic features. 

## Table of Contents 
1. 01_Executive_Summary
2. 02_Data_Cleaning_Imputation 
3. 03_EDA
4. 04_Modeling 

Link to Executive Summary: https://github.com/jguo052/Project-4/blob/main/code/01_Executive_Summary.md

The data used in this project was sourced from the US Environmental Protection Agency's Environmental Justice Mapping and Screening Tool (EJSCREEN). The dataset gives information on various environmental and demographic features for census blocks and tracts in the United States in 2021. 

Link to data: https://gaftp.epa.gov/EJSCREEN/2021/
From this link, the file the project uses is EJSCREEN_2021_StatePctile_Tracts.csv, the 9th link down on the page. The other files on the page are the dataset aggregated across different geographic regions, and there are two links for each dataset saved in different formats. The 9th link will open up our dataset in a CSV format.

Data was downloaded as a CSV file from the Environmental Protection Agency website. It was read into a dataset using Pandas software. The data file was larger than the limit GitHub allows to be uploaded, so Git Large File Storage was used to create a pointer file that could be uploaded and used as a map to find the large file. In order to download the file and run the notebooks, users need to have Git Large File Storage installed on their computer. Instructions for installing Git LFS can be found here: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

Data cleaning involved dropping redundant columns, removing rows with all 0 or null values, removing rows missing the target variable, and imputing values for the remaining null values in our data. Observations with a population lower than 30 were dropped, and observations that had a significant number of zeros that seemed unlikely across multiple features were also dropped. A major outlier in our target variable was dropped. Region, a categorical variable, was OneHotEncoded to prepare the data for modeling.  

## Software Requirements
This analysis uses Pandas and Numpy to work with dataframe data. For visualizations, this analysis uses MatPlotLib and Seaborn. Preprocessing and modeling was done using packages from Sklearn.
