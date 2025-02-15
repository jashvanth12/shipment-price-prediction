# shipment-price-prediction           

# Business Context
A company sells sculptures acquired from various artists worldwide. These sculptures vary in size, weight, material, and destination, leading to fluctuating shipping costs. Accurately predicting shipping costs is essential for pricing strategies, cost optimization, and improving customer satisfaction.
# Objective
The goal is to develop a machine learning model that predicts the shipping cost of sculptures based on key attributes such as sculpture dimensions, weight, material, shipping method, and destination.

# Challenges
  * Variability in Sculpture Characteristics: Different sizes, weights, and materials impact packaging and handling.
  * Diverse Shipping Methods: Costs vary based on air, sea, or land transport.
  * Geographical Factors: Distance, customs fees, and taxes influence the final cost.
  * Insurance & Fragility Considerations: Fragile or expensive sculptures may require special packaging and insurance.

# Summary of the dataset
![image](https://github.com/user-attachments/assets/bf626fff-52d5-460b-b177-7536aba2ef92)

### Statistical Inferences
  * Median Price Of Sculpture is 1192.
  * 25% of the population is Price Of Sculpture below 5
  * Average Artist Reputation of the population is 0.46
  * The difference between 75th percentile and Max also suggests the skewness 
  * We can confirm the insights we got from data distribution, as skewness is more for Height, Width, and Base Shipping Price, and skewness of Weight, Price of Sculpture because of Outliers

# Check For Data Types
![image](https://github.com/user-attachments/assets/936dd38d-449e-4f71-a2c9-9e5b182484c0)

**Most of the features seems to be of object data type, Now let's seperate Categorical and numerical columns**

# Univariate Analysis
### Numerical Features
![image](https://github.com/user-attachments/assets/69fb6dcd-e0cf-405b-8fdc-a4769ddcf653)

* Height, Width and Base Shipping price are positively skewed
* Weight, Price of Sculpture has many outliers

### Categorical Features
![image](https://github.com/user-attachments/assets/c7b13214-60dc-49e2-a956-a05ef8c23bce)

* Customer Id, Artist Name, Customer location have 6500 unique values, so they can be dropped
* Scheduled date and Delivery date needs feature engineering

![image](https://github.com/user-attachments/assets/becdc23a-e821-44e5-bb3e-afa57eb32c0b)

* Material column` has seven unique value, which are almost equally distributed 
* There are 6 bi-variate categorical columns

# Checking Null Values In The Dataset
![image](https://github.com/user-attachments/assets/3cca5f1e-5bae-4f18-b6c2-731b5dc6a40d)

* There are 7 columns which has null values

# Multivariate Analysis
correlation in numerical Features
![image](https://github.com/user-attachments/assets/3efadba8-a6e9-4428-bb4c-9dc731a78aa5)

* There is a high correlation between Height-Width, and Weight-Price of Sculpture

# Relation between target and numerical features
![image](https://github.com/user-attachments/assets/d9deafc1-c040-4169-a2a8-e6b121537246)

 *  We can observe that there seems to be very low linear relationship between the independent and dependent features
 * There seems to be linear relationship between Price of sculpture and the target column

## Visualizing Independent columns
### Target Feature
![image](https://github.com/user-attachments/assets/5472c1ca-0b9b-4f2d-97cf-b583c5cf3261)

![image](https://github.com/user-attachments/assets/d64ee667-ca3b-4cc9-92dc-bbaada34b5c4)

* There are outliers in the target feature and we need to transform

### How shipment type is affecting cost of shipment?
![image](https://github.com/user-attachments/assets/626e03d6-e6ab-41fe-accb-d6cadda304e4)

 * There is only a difference of 43 in the shipping cost between International and domestic, so it won't affect much at the prediction
 * There are much more domestic shipement than international

### How Express Shipment is affecting cost of shipment?
![image](https://github.com/user-attachments/assets/5d3a0a0d-ef86-4bf0-ba76-6c3b0692b5c1)

 * In express shipment also, there is only a difference of 83 in the shipping cost between International and domestic, so it won't affect much at the prediction
 * It's obvious that there are more normal delivery than express

### How Installation included or not is affecting cost of shipment?
![image](https://github.com/user-attachments/assets/b6701e13-77c7-4f1b-8be6-b96324e01215)

 * There is only a difference of 53 which won't make much a difference at the time of prediction
 * We can see that the charges for not installation is less and values are more, which says that a business should try to take contract which requires installation

### How Fragile Cost is affecting cost of shipment?
![image](https://github.com/user-attachments/assets/46b5bfa4-643a-4e37-9242-5343d2da61ee)

 * There is only a difference of -80 which says that the median cost is more if there is no fragile cost and because the number is small it won't make much of a difference at the time of training model

### How delivery location is affecting cost of shipment?
![image](https://github.com/user-attachments/assets/a09ae2b4-c4ec-41e4-9983-c12f832604c3)

 * There is only a difference of 27 which won't make much a difference at the time of prediction

### How Customer Financial condition is affecting cost of shipment?
![image](https://github.com/user-attachments/assets/94e1ec94-8856-4ee6-8dbc-ab31c126b612)

* There is only a difference of 66 which won't make much a difference at the time of prediction
* After understanding relationship between the target column and categorical columns, that there doesn't seem to be much pattern, which model can find in the categorical columns also

# Visualizing date columns
![image](https://github.com/user-attachments/assets/4c0edf09-8454-4c22-abbb-62f56f676736)

![image](https://github.com/user-attachments/assets/0ddcfc66-e661-4f26-b3bd-7a0075c938e8)

![image](https://github.com/user-attachments/assets/a80dc2c9-a06d-4836-a90c-946e69eef726)

  * Monthly distribution of every year is different
  * For 2015, 2016 and 2018 There were more cost at the start and end of the year.
  * For 2017 and 2019 cost were high mid year

![image](https://github.com/user-attachments/assets/2bdc6672-8943-4568-b0a4-3ef713b917e6)

![image](https://github.com/user-attachments/assets/e40a9649-6b22-406d-a872-5a8d2c592f9a)

### Final 
 * The Cost column is the target to predict.
 * The target variable here is continuous.
 * There are outliers in some columns we have to remove outliers.
 * `date` column should be configured to extract `year` and `month`.
 * Null values in `Artist Reputation`, `Height`, `Width`, `Weight`, `Material`, `Transport`, `Remote Location` needs to be handled

# Data Cleaning
### Check Null Values
![image](https://github.com/user-attachments/assets/300d97c6-baa7-4cbe-a49f-6ec87cbc3c49)

 * Imputed Null Values with Median

### Checking Duplicate Values
  * There are no duplicates values in the data

### Multicollinearity Check
#### Variance Inflation Factor (VIF)
![image](https://github.com/user-attachments/assets/6969c91a-ff43-47b1-aefb-2ef831700be1)

 * Note:  All the columns has VIF around 1, so there is not severe multicollinearity

![image](https://github.com/user-attachments/assets/2576e79b-b596-4a02-9761-99cffd55ec89)

   - From EDA we have seen that the `Price Of Sculpture ` and `Weight` were extremely positively skewed may be because outliers
   - So we can handle this type of case using Power Transformer or log transformer
   - `Width` and `Base Shipping Price` too are slighty positively skewed but not severe as shown by skewness coefficient.

#### Checking the power transformer on outlier columns
![image](https://github.com/user-attachments/assets/0cc0af01-a15a-4c71-b4ee-da19e0121dba)

  -  Power transformer does a better job at overcoming skewness and handling outliers

#### Removing outliers from Target Column
* Because most model would perform better on noiseless data, as Outlier might skew the findings of your model in one direction

  ![image](https://github.com/user-attachments/assets/5107151d-f000-4e5c-b2e9-4f1e4410f660)

  Need to apply transformation to target column to make it normally distributed. we use 1p to avoid `inf` values arise dued to `0` in the target column

![image](https://github.com/user-attachments/assets/aac9ab92-3165-42c1-9135-c59511bd1370)

Data looks good after the log transformation but not Normally distributed.

#### Feature Creation for Date Columns
![image](https://github.com/user-attachments/assets/6fbb1a23-8d7e-4d93-ba24-f483e83b7c2a)

### Drop Unnessary Columns
- Drop `Customer Id`, `Artist Name` and `Customer Location` columns having huge number of unique values.
- `Scheduled Date` and `Delivery Date` can be dropped since the required features are extracted form it

# MOdel Building
![image](https://github.com/user-attachments/assets/afb4bddd-fd2b-464e-9aae-d69e423a1627)

![image](https://github.com/user-attachments/assets/5cdf710f-74e9-4fae-a9c7-1f63bb64f3a4)

# Model Selection
![image](https://github.com/user-attachments/assets/ff33673b-61f1-4d6f-a26f-d73e82ba19b8)

# Hyoerperameter Tuning
![image](https://github.com/user-attachments/assets/b96d0654-b15b-4b68-92ae-93e0f8b0dacf)

 ![image](https://github.com/user-attachments/assets/76b2a11d-9c9a-439b-88cf-942c4da00dd4)

 ![image](https://github.com/user-attachments/assets/5032f934-4ba7-478f-8e04-fe7037f3bff7)

 ### Best Model is CATBoost Regressor with 97.17% Score





















