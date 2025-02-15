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

