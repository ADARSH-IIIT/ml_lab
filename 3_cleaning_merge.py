import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Simulate the datasets
# ===============================

# Orders dataset
orders_data = {
    "OrderID": [1,2,3,4,5,6,7,8,9],
    "CustomerID": ["C001","C002","C003","C002","C004","C001","C005","C003","C004"],
    "Product": ["Laptop","Mouse","Keyboard","Monitor","Laptop","Mouse","Headset","Laptop","Monitor"],
    "Quantity": [1,2,1,1,1,3,2,1,2],
    "Price": [70000,1200,2500,np.nan,72000,1100,3000,71000,15000],
    "OrderDate": ["2021-01-05","2021-02-10","2021-03-15","2021-04-02","2021-05-18",
                  "2021-06-22","2021-07-30","2021-08-12","2021-09-05"]
}
orders = pd.DataFrame(orders_data)

# Customers dataset
customers_data = {
    "CustomerID": ["C001","C002","C003","C004","C005"],
    "Name": ["Alice","Bob","Charlie","Diana","Ethan"],
    "City": ["Delhi","Mumbai","Bangalore","Delhi","Chennai"],
    "SignupDate": ["2020-11-15","2021-01-10","2020-12-01","2021-03-20","2021-02-18"]
}
customers = pd.DataFrame(customers_data)

# Convert dates
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])
customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])





# ===============================
# 2. Handle missing values (NO LAMBDA)
# ===============================

def fill_missing_price(group):
    mean_value = group["Price"].mean()
    group["Price"] = group["Price"].fillna(mean_value)
    return group

orders = orders.groupby("Product").apply(fill_missing_price)

# group by product  create smaller df of same product name  ,    assume them same df smaller 
#  group by product , city , now ab dono ka same toh unje ek enetity consider karege 

# Product   Price
# Apple     100
# Apple     110
# Apple     120
# Banana    50
# Banana    45
# Banana    40









# ===============================
# 3. Add TotalAmount column
# ===============================
orders["TotalAmount"] = orders["Quantity"] * orders["Price"]





# ===============================
# 4. Merge datasets
# ===============================
merged = pd.merge(orders, customers, on="CustomerID")







# ===============================
# 5. Total revenue from each city
# ===============================
revenue_per_city = merged.groupby("City")["TotalAmount"].sum()





# ===============================
# 6. Customer who spent maximum
# ===============================
# making small small groups of same name , then add them ac to toalsum and pick the max , lofigccaly we are spltting df acc to name , then asdding them , and picking hte maxsum amonfg them 
customer_max_spent = merged.groupby("Name")["TotalAmount"].sum().idxmax()






# ===============================
# 7. Most popular product (by quantity)
# ===============================
most_popular_product = merged.groupby("Product")["Quantity"].sum().idxmax()




# ===============================
# 8. Monthly revenue trend , creating a new having year-month only , then clubbing htem which has same year month 
# ===============================
merged["Month"] = merged["OrderDate"].dt.to_period("M")
monthly_revenue = merged.groupby("Month")["TotalAmount"].sum()







# ===============================
# 9. Top 2 customers by revenue in each city
# ===============================

#  reset index , so that the forst col city do not become the index col
top2_customers_city = merged.groupby(["City","Name"])["TotalAmount"].sum().reset_index()
top2_customers_city = top2_customers_city.sort_values(["City","TotalAmount"], ascending=[True,False])
top2_customers_city = top2_customers_city.groupby("City").head(2)



# df1
# A  B    23
# A  B     34
# A  B     100

# A  C    50
# A  C    60










# ===============================
# 10. Visualizations (Blocking)
# ===============================

# Bar chart: revenue per city
revenue_per_city.plot(kind="bar", title="Revenue per City")
plt.ylabel("Revenue")
plt.show()  # BLOCKING

# Line chart: monthly revenue trend
monthly_revenue.plot(kind="line", marker="o", title="Monthly Revenue Trend")
plt.ylabel("Revenue")
plt.show()  # BLOCKING

# Pie chart: product-wise revenue share
product_revenue = merged.groupby("Product")["TotalAmount"].sum()
product_revenue.plot(kind="pie", autopct="%1.1f%%", title="Product-wise Revenue Share")
plt.ylabel("")
plt.show()  # BLOCKING


# ===============================
# Print results
# ===============================
print("\nTotal revenue per city:\n", revenue_per_city)
print("\nCustomer who spent maximum:", customer_max_spent)
print("\nMost popular product:", most_popular_product)
print("\nMonthly revenue trend:\n", monthly_revenue)
print("\nTop 2 customers by revenue in each city:\n", top2_customers_city)





