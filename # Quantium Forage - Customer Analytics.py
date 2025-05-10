# Quantium Forage - Customer Analytics Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')

# Load datasets
transactions = pd.read_csv('/kaggle/input/quantium-data-analytics-virtual-experience-program/Transactions.csv')
purchase_behaviour = pd.read_csv('/kaggle/input/quantium-data-analytics-virtual-experience-program/PurchaseBehaviour.csv')

# Merge datasets on LYLTY_CARD_NBR
df = transactions.join(purchase_behaviour.set_index('LYLTY_CARD_NBR'), on='LYLTY_CARD_NBR')
df['DATE'] = pd.to_datetime(df['DATE'], origin='1899-12-30', unit='D')

# Remove outliers
outliers = df[(df['PROD_QTY'] == 200) | (df['TOT_SALES'] > 600)].index
df.drop(index=outliers, inplace=True)

# Extract brand and packet size using regex
df['BRAND'] = df['PROD_NAME'].str.extract(r'(\b[A-Z][a-zA-Z]*\b)')
df['BRAND'] = df['BRAND'].replace({
    'RRD': 'Red', 'NCC': 'Natural', 'Dorito': 'Doritos', 'WW': 'Woolworths',
    'Grain': 'GrnWves', 'Infzns': 'Infuzions', 'Snbts': 'Sunbites', 'Smith': 'Smiths'
})
df['PKT_SIZE'] = df['PROD_NAME'].str.extract(r'(\d+[gG])')

# Filter for chip-related products
chips_df = df[df['PROD_NAME'].str.contains(r'Chips|Chip|Chp', case=False, na=False)]

# Total sales by Premium Customer
premium_sales = chips_df.groupby('PREMIUM_CUSTOMER')['TOT_SALES'].sum()
plt.pie(premium_sales, labels=premium_sales.index, explode=[0,0.03,0], autopct='%1.1f%%')
plt.title('Total Sales by Premium Customer')
plt.show()

# Sales by Lifestage
lifestage_sales = chips_df.groupby('LIFESTAGE')['TOT_SALES'].sum()
plt.figure(figsize=(10,5))
sns.barplot(x=lifestage_sales.index, y=lifestage_sales.values)
plt.xticks(rotation=90)
plt.title('Total Sales by Lifestage')
plt.show()

# Sales by Premium Customer and Lifestage
combined_sales = chips_df.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE'])['TOT_SALES'].sum()
plt.figure(figsize=(10,5))
sns.barplot(x=combined_sales.index.get_level_values('LIFESTAGE'),
            y=combined_sales.values,
            hue=combined_sales.index.get_level_values('PREMIUM_CUSTOMER'))
plt.xticks(rotation=90)
plt.title('Total Sales by Customer Type and Lifestage')
plt.show()

# Quantity analysis
quantity_by_customer = chips_df.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE'])['PROD_QTY'].sum()
plt.figure(figsize=(10,5))
sns.barplot(x=quantity_by_customer.index.get_level_values('LIFESTAGE'),
            y=quantity_by_customer.values,
            hue=quantity_by_customer.index.get_level_values('PREMIUM_CUSTOMER'))
plt.xticks(rotation=90)
plt.title('Total Quantity by Customer Type and Lifestage')
plt.show()

# Average quantity per customer
customer_summary = chips_df.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).agg(
    total_quantity=('PROD_QTY', 'sum'),
    total_customers=('LYLTY_CARD_NBR', 'nunique')
)
customer_summary['avg_quantity'] = customer_summary['total_quantity'] / customer_summary['total_customers']

# Top selling brands
brand_sales = chips_df.groupby('BRAND')['TOT_SALES'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,4))
sns.barplot(x=brand_sales.index, y=brand_sales.values)
plt.title('Brand vs Total Sales')
plt.show()

# Most popular packet sizes
pkt_size_count = chips_df['PKT_SIZE'].value_counts()
plt.figure(figsize=(6,4))
sns.barplot(x=pkt_size_count.index, y=pkt_size_count.values)
plt.title('Most Purchased Packet Sizes')
plt.show()

# Correlation between PKT_SIZE, PROD_QTY, and TOT_SALES
pkt_size_numeric = chips_df['PKT_SIZE'].str.extract(r'(\d+)').astype(int)
corr_df = pd.concat([pkt_size_numeric, chips_df[['PROD_QTY', 'TOT_SALES']].reset_index(drop=True)], axis=1)
corr_df.columns = ['PKT_SIZE', 'PROD_QTY', 'TOT_SALES']
plt.figure(figsize=(6,4))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Monthly trends
monthly_sales = chips_df.groupby(pd.Grouper(key='DATE', freq='M'))['TOT_SALES'].sum()
plt.figure(figsize=(15,4))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Monthly Sales Trend')
plt.show()

# Monthly sales by premium customer
monthly_premium_sales = chips_df.groupby([pd.Grouper(key='DATE', freq='M'), 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum().unstack()
monthly_premium_sales.plot(kind='bar', stacked=True, figsize=(20,6))
plt.title('Monthly Sales by Premium Customer')
plt.ylabel('Total Sales')
plt.show()

# Monthly quantity by lifestage
monthly_lifestage_qty = chips_df.groupby([pd.Grouper(key='DATE', freq='M'), 'LIFESTAGE'])['PROD_QTY'].sum().unstack()
monthly_lifestage_qty.plot(kind='bar', stacked=True, figsize=(20,6))
plt.title('Monthly Quantity by Lifestage')
plt.ylabel('Product Quantity')
plt.show()