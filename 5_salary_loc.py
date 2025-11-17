import pandas as pd
import numpy as np

# ===============================
# 1. Create list of employees
# ===============================

names = ["Alice", "Bob", "Charlie", "David", "Emma"]
ages = [24, 27, "Not Available", 32, 29]
departments = ["HR", "IT", "Finance", "IT", "Unknown"]
salaries = [40000, -1, 35000, 52000, 45000]

employees = []

for i in range(len(names)):
    emp = {
        "Name": names[i],
        "Age": ages[i],
        "Department": departments[i],
        "Salary": salaries[i]
    }
    employees.append(emp)

# ===============================
# 2. Create DataFrame
# ===============================
df = pd.DataFrame(employees)
print("Initial DataFrame:\n", df)

# ===============================
# 3. Show first 3 rows, shape, columns
# ===============================
print("\nFirst 3 rows:\n", df.head(3))
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

# ===============================
# 4. Add 10% bonus
# ===============================
df["Bonus"] = df["Salary"] * 0.10
print("\nDataFrame with Bonus column:\n", df)

# ===============================
# 5. HR salary increased by 5%
# ===============================
df.loc[df["Department"] == "HR", "Salary"] *= 1.05
print("\nAfter HR salary increase:\n", df)

# ===============================
# 6. Employees earning more than 45,000
# ===============================
df_highSalary = df[df["Salary"] > 45000]
print("\nEmployees earning more than 45,000:\n", df_highSalary)

# ===============================
# 7. Replace invalid values with NaN
# ===============================
df.replace({
    "Age": "Not Available",
    "Department": "Unknown"
}, np.nan, inplace=True)

df.loc[df["Salary"] < 0, "Salary"] = np.nan
df.loc[df["Bonus"] < 0, "Bonus"] = np.nan

print("\nAfter replacing invalid values with NaN:\n", df)

# ===============================
# 8. Forward Fill & Backward Fill
# ===============================
df_bfill = df.bfill()
df_ffill = df.ffill()

print("\nBackward Fill:\n", df_bfill)
print("\nForward Fill:\n", df_ffill)


