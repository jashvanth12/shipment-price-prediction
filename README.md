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

