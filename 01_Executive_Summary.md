# Executive Summary 
## Problem Statement
A variety of characteristics can contribute to increased cancer rates, including environmental influences and demographic characteristics. Using the Environmental Justice Screen data provided by the United States Environmental Protection agency, we want to build regression models to predict the air toxics cancer risk per million person in each census block group using other features such as demographic information and environmental features. Additionally, we want to compare the performance of those models to a model built using only environmental feature data. This is informed by prior knowledge that low income areas and communities of color often are located near environmental hazards and have increased vulnerability to negative effects such as greater cancer risk.
Target audience: These predictions are being generated for the United States government to aid in budgeting for healthcare costs for Medicare and Medicaid for different areas in the United States.

## Description of Data
The data used in this project was sourced from the US Environmental Protection Agency's Environmental Justice Mapping and Screening Tool (EJSCREEN). The dataset gives information on various environmental and demographic features for census blocks and tracts in the United States in 2021. The dataset has 24 variables and 73,124 observations. 

## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|st_name|object|cleaned_ejscreen| State name| 
|st_abbrev|object|cleaned_ejscreen| State abbreviation| 
|region|integer|cleaned_ejscreen|Region number based on the Environmental Protection Agency (EPA) regions determinations, in models the region | 
|pop|integer|cleaned_ejscreen|| 
|poc_pct|float|cleaned_ejscreen|Percent people of color| 
|low_inc_pct|float|cleaned_ejscreen|Percent low income, defined as household income that is less than or equal to twice the federal poverty level.| 
|sub_hs_pct|float|cleaned_ejscreen| Percent with less than high school education| 
|ling_iso_pct|float|cleaned_ejscreen|Percent of linguistically isolated households, in which all members age 14 years and over speak a nonEnglish language and also speak English less than “very well”| 
|under_5_pct|float|cleaned_ejscreen| Percent of people under 5 years old| 
|over_64_pct|float|cleaned_ejscreen| Percent of people over 64 years old| 
|unemp_pct|float|cleaned_ejscreen| Percent unemployed| 
|pre_1960_pct|float|cleaned_ejscreen|Percent of housing units built pre-1960, as indicator of potential lead paint exposure  | 
|diesel_pm|float|cleaned_ejscreen|Diesel particulate matter level in air, µg/m3| 
|cancer|float|cleaned_ejscreen| Lifetime cancer risk from inhalation of air toxics, per million people| 
|resp_index|float|cleaned_ejscreen|Air toxics respiratory hazard index (ratio of exposure concentration to health-based reference concentration)| 
|traffic_prox|float|cleaned_ejscreen|Count of vehicles (AADT, avg. annual daily traffic) at major roads within 500 meters, divided by distance in meters| 
|npl_prox|float|cleaned_ejscreen|Count of proposed and listed NPL sites within 5 km (or nearest one beyond 5 km),each divided by distance in kilometers. National Priorities List (NPL) is the list of sites of national priority among the known releases or threatened releases of hazardous substances, pollutants, or contaminants.| 
|rmp_prox|float|cleaned_ejscreen| Count of RMP (potential chemical accident management plan) facilities within 5 km (or nearest one beyond 5 km), each divided by distance in kilometers| 
|waste_prox|float|cleaned_ejscreen|Count of TSDFs (hazardous waste management facilities) within 5 km (or nearest beyond 5 km), each divided by distance in kilometers| 
|ozone|float|cleaned_ejscreen|Ozone summer seasonal avg. of daily maximum 8-hour concentration in air in parts per billion| 
|pm_25|float|cleaned_ejscreen|Annual average of particulate matter levels in air, µg/m3| 
|undgrd_stor|float|cleaned_ejscreen|Underground storage tanks|
## Data Cleaning and Null Value Imputation 

From the raw data, columns beginning with B_ and T_ that provided information for creating specialized images were dropped, as they were not necessary for our exploratory data analysis. Columns beginning with P_ and D_  were also dropped, as they  contained percentile and index data, which was representing the same features as other columns, just in a different format. Because these could be reconstructed from other columns and were representing the same information, we felt comfortable dropping them. We also dropped the column dem_index further along in our data cleaning process when we realized this column was just the product of the poc_pct and low_inc_pct columns. We renamed our columns to be more descriptive of the variable they represented, made column names lowercase, and replaced spaces with underscores.

We then dropped all rows where the entire observation consisted of zero or null values. We dropped rows where our target value, cancer, was missing, as the variable we wanted to predict was an especially important variable for our modeling. There were only 246 rows missing a value for cancer, which was less than 1% of our data. Before dropping these rows, we checked to see if the missing values seemed random, to ensure there wasn't a pattern to the missing data. The rows with missing cancer data were relatively evenly distributed across the states, so there didn't seem to be a pattern to the null values. We found that the majority of rows missing cancer data had a population of zero. We felt comfortable dropping these rows because a population of zero clearly indicates an error in data collection. When dropping rows missing cancer data, we ended up dropping all the rows with missing diesel_pm and resp_index values, as the rows corresponded. 

The variable waste_water had 19,401 null values. This column seemed difficult to salvage, even with imputation for missing values. Additionally, this column seemed less likely to be related to our target variable of air toxics cancer rate, as it wasn't related to air pollution. Therefore, we decided to drop this column entirely.  

When looking at our columns that were measured in percent of population, we realized that our population sizes varied largely across observations; therefore a percentage of 100% in an observation with a population of 20 means something different than a percentage of 100% in an observation with a population of 5000. When looking into the reason that some observations had such a low population, we realized that while most observations represent US Census tracts, which generally have a population size above 1200, observations with lower populations represented tribal data. We did not want to completely remove tribal data, especially due to the history of environmental injustice towards indigenous people. However, we did decide to drop rows with a population under 30. This preserves much of the tribal data while allowing us more confidence in the interpretation of features measured in percent of population. 

We then looked at rows with many 0 values. While it is possible for many of this features to have accurate measurements that were 0, there were also entries of 0 that seemed like they may have been placeholders for lack of data on that feature. For example, a Census tract containing six values of 0, including values of 0 for unemployment percent, percent under 5, and percent over 64, had a population of 6,372. It is hard to imagine that there are no unemployed or elderly people in an observation with such a high population. Unfortunately, because we cannot accurately determine when these zeros are genuine zeros or missing values, we did not have a great way to correct this problem. We decided to remove rows with 5 or more 0 values. 

Next, we noticed a major outlier of 2000 in our target variable, cancer. The outlier was from Census tract data from Puerto Rico. with high POC and low income percentages, but otherwise typical values in the other columns. After further research, we were unable to find any indication that areas in Puerto Rico have extremely high instances of cancer, and therefore it is possible that this observation is an error. We decided to this outlier. 

The next largest cancer rate we see in the data is a value of 1,000 that comes from a Census tract in Louisiana. There is an infamous area in Lousiana known as ‘Cancer Alley’ with extraordinarily high cancer rates. Therefore this observation may be reasonable to leave in our dataset.

We considered dropping the remaining null values from our dataset, but realized that there was a pattern to the missing data. Dropping the null values would completely remove observations from Puerto Rico, Hawaii, and Alaska from our dataset. To address these remaining null values, we used iterative imputer with a linear regression as our estimator to impute values for the remaining null values. 


## Analysis
#### Exploratory Data Analysis
A few operations were performed as an intial exploration of the subreddit data. 

First, a bar chart was made to represent the frequency of observations of each state. The bars were color coded based on the region the state was in. 

<img src='save bar chart and put it here'>

Next, box plots were made, separated by region, for each of the environmental and demographic features in the data set to visualize the spread of data for each feature. These box plots helped to visualize outliers and compare the spread of data in different geographic regions of the United States.

<img src='save regions box plot and put it here'>

explain bar charts 

Boxplots were also made for each of the features without grouping them by region. These box plots helped visualize the overall spread of the feature and the presence of outliers calculated with the mean and standard deviation of the entire data for that feature. 

<img src='save overall box plot'>

Histograms were also made for each of the features. These histograms helped visualize the spread of the data and identify skew. For example, in the histogram of the distribution of data for our target variable, cancer, the distribution had a significant right skew. This was important to note because later we were able adjust for that skew in our linear modeling by applying a log transformation to our target variable. 

<img src='save overall hist and put it here'>

Next, LINE assumptions were checked to see if, once we built a linear regression model, the coefficients of that model could be interpreted for inference. 

The LINE assumptions are:
1. Linear relationship between features and target variable 
2. Independence of observations
3. Normality of residuals
4. Equal variance of residuals 

If the LINE assumptions are violated, the model can still be used for prediction, but cannot be used for inference. 

To check for a linear relationship, scatter plots between each feature and the target variable were plotted. Each of the scatter plots did not show a linear relationship. Points were concentrated near the x-axis with a few points floating above the rest. There was no linear pattern to the plot. For example, the scatter plot below shows the relation between _________ and cancer. This is representative of the scatter plots between each of the features and the target variable. Our data does not meet the assumption of linear relationship between features and target variables.

<img src='save linear scatter and put it here'>

To check for independence of observations, the Variance Inflation Factor (VIF) score was calculated for each of the features. The VIF score is a measure of how well a feature is explained by the other features in the data set. A high VIF score indicates multicollinearity, with a VIF score above 5 considered in violation of the assumption of independence of observations. Ten of our features had VIF scores above 5, with some features (pm_25 and ozone) having VIF scores as high as 28 and 33. Our data does not meet the assumption of independence of observations. 

To check for normality of residuals, a linear regression model was built. Predictions were made with the model, and residuals were calculated by subtracting the prediction from the true y values. A histogram of the residuals was plotted to check for normality. The residuals had a substantial right skew. Our data does not meet the assumption of normality of residuals. 

<img src='save resids hist and put it here'>

To check for the equal variance of residuals, a scatter plot of the relation between the residuals and the predictions was plotted. If the residuals had equal variance, the points on the scatterplot would be randomly distributed and show no pattern. However, in this plot the residuals clustered near the bottom of the plot and were not randomly distributed. Our data does not meet the assumption of equal variance of residuals. 


<img src='save resids scatterplot and put it here'>



# Modeling

We built models to predict air toxic cancer rates using two different sets of features. One of our sets was comprised of all the environmental features in the dataset. The second set included all the demographic features in addition to the environmental features. We wanted to see if models including those demographic features would perform better than models with only environmental features. This was informed by prior knowledge that low income areas and communities of color often are located near environmental hazards and have increased vulnerability to negative effects such as greater cancer risk.

We fit ten different models on both sets of features to find which model and features best performed at predicting air toxics cancer rates. Prior to fitting our models, we split our data into a training set and a validation set so that we would be able to see how well our model performs on new data after fitting on the training data. 

### Linear Regression

#### Environmental and Demographic Features 
We noticed during EDA that our target variable, cancer, had a substantial right skew in the distribution of data. For the linear regression, a log was performed on both y train and y val to try and normalize the distribution and build a stronger model. 

For our linear regression model, we built a pipeline with 


so we could gridsearch for the best hyperparameters. We used GridSeach to check whether CountVectorizer stop words should be none or english, whether CountVectorizer ngram_range should be only single words or single words and word pairs, and whether CountVectorizer max_df should be 0.9 or 1.0. We also GridSearched over AdaBoost Classifier to check whether n_estimators should be 100, 125, or 150, and whether the learning rate that weights incorrect classifications each iteration of the model should be 1.0 or 1.5. The GridSearch returns the combination hyperparameters that built the best performing model. The AdaBoost best parameters were: 'cvec__max_df': 0.9, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': None, 'ada__learning_rate': 1.0, 'ada__n_estimators': 150. 



#### Environmental Features Only

### Linear Regression with Lasso Regularization
#### Environmental and Demographic Features 

#### Environmental Features Only
For our AdaBoost Classifier, we built a pipeline with CountVectorizer and AdaBoost Classifier so we could gridsearch for the best hyperparameters. We used GridSeach to check whether CountVectorizer stop words should be none or english, whether CountVectorizer ngram_range should be only single words or single words and word pairs, and whether CountVectorizer max_df should be 0.9 or 1.0. We also GridSearched over AdaBoost Classifier to check whether n_estimators should be 100, 125, or 150, and whether the learning rate that weights incorrect classifications each iteration of the model should be 1.0 or 1.5. The GridSearch returns the combination hyperparameters that built the best performing model. The AdaBoost best parameters were: 'cvec__max_df': 0.9, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': None, 'ada__learning_rate': 1.0, 'ada__n_estimators': 150. 

    
We then fit this model on our X_train data. This model had an accuracy score of 0.93 on the train data and 0.89 on the test data. On test data, the model sensitivity was 0.89, the model specificity was 0.89, and the model precision was 0.86. This model performed almost as well as the Logistic Regression model and had less variance. 


To visualize the performance of our models, we created confusion matrix plots that displayed the counts of true positives, true negatives, false positives, and false negatives. These visualizations helped us see which models were peforming best, as well as see what types of posts some models were struggling at classifying.


<img src='visualizations /adacm.jpg'>

The confusion matrix for the AdaBoost model shows that the model did well classifying submissions with few false positives or false negatives. 

### Linear Regression with Ridge Regularization
#### Environmental and Demographic Features 
#### Environmental Features Only

### Linear Regression with ElasticNet Regularization
#### Environmental and Demographic Features 
#### Environmental Features Only

### K Nearest Neighbors
#### Environmental and Demographic Features 
#### Environmental Features Only

### Decision Tree
#### Environmental and Demographic Features 
#### Environmental Features Only

### Bagging Regressor
#### Environmental and Demographic Features 
#### Environmental Features Only

### Random Forest
#### Environmental and Demographic Features 
#### Environmental Features Only

### AdaBoost
#### Environmental and Demographic Features 
#### Environmental Features Only

### Gradient Boost
#### Environmental and Demographic Features 
#### Environmental Features Only






## Conclusions and Recommendations 

We were able to definitively answer our problems statement, showing that by analyzing text data from the Religion and AskPhilosophy subreddits using natural language processing and classification models, a machine learning model can predict which subreddit the text data came from with high accuracy. We recommend using our logistic regression model, which  was able to classify submisssions with 91% accuracy, a sensitivity of 0.91, a specificity of 0.90, and a precision of 0.87. One downside of this logistic regression model compared to the next best performing model, AdaBoost, is the higher variance that this model had, about a 0.10 decrease in accuracy from the train data to the test data. The type of regularization, as well as the degree of penalization, could be explored with this model to see if it decreases the variance. 

Even without key words like 'religion' and 'philosophy', this logistic regression model was able to pick up on differences in text from the two subreddits and classify submissions with 90% accuracy. 

When misclassifications occured, investigating instances of misclassification and the corresponding text showed that the misclassified posts often contained words common in one subreddit used in a submission on the other subreddit, or did not have much relation to the topic of either subreddit, confusing the model. It makes sense that classification errors may occur with these types of posts. 

The linear regression models using CountVectorizer to transform the text data are strong models that can be used for classifying whether a submission came from r/Religion or r/AskPhilosophy. 

