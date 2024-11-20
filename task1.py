import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('datas/dataset.csv')
print(df.shape)
print(df.columns)

# Plot 1: Distribution of churn in the dataset
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='coolwarm')
plt.title('Customer Churn Distribution')
plt.show()

# Plot 2: Distribution of tenure for customers who churned vs. those who didn't
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30, palette='coolwarm')
plt.title('Tenure Distribution by Churn Status')
plt.show()

# Plot 3: Relationship between MonthlyCharges and Churn
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='coolwarm')
plt.title('Monthly Charges vs. Churn')
plt.show()

#Observation - 1. Churn Rate: A good portion of customers are leaving the service (about 27%). 
#                             While most customers stay, it's important to focus on why some are leaving to improve retention.
#              2. Tenure (How long customers stay): Customers who have been with the company for a short time (less than a year) are more 
#                                                   likely to leave. Those who have stayed longer (over 3 years) tend to remain loyal.
#              3. Monthly Charges: Customers with higher monthly bills are more likely to leave than those with lower bills. 
#                                  This shows that cost might be pushing people to leave.