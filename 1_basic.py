import pandas as pd
import numpy as np
from datasets import load_dataset   # pip install datasets

# -----------------------------------------
# Creating a DataFrame from a dictionary
# -----------------------------------------
data = {"Name":["Alice", "Bob", "David"], 
        "id": ["mtp01", "mtp03", "mtp02"], 
        "subject": ["MFDS", "ML", "NLP"]}

df = pd.DataFrame(data)
print(df)


print("\n\n\n\n")


# -----------------------------------------
# Creating a DataFrame from a list of lists
# -----------------------------------------
data_L = [
    ["Alice", "mtp01", "MFDS"], 
    ["Bob", "mtp02", "ML"], 
    ["David", "mtp03", "NLP"]
]
df_L = pd.DataFrame(data_L, columns=["name", "roll_no", "subject"])
print(df_L)
print("\n\n\n\n")




# -----------------------------------------
# Creating a DataFrame from list of dictionaries
# -----------------------------------------
data_d = [
    {"name":"Alice", "roll_no":"mtp01", "subject":"MFDS"},
    {"name":"Bob", "roll_no":"mtp02", "subject":"ML"},
    {"name":"David", "roll_no":"mtp03", "subject":"NLP"}
]
df = pd.DataFrame(data_d)
print(df)
print("\n\n\n\n")




# ---------------------------------------------------------
# REPLACE THIS WITH YOUR LOCAL CSV PATH
# Example: "/home/adarsh/Desktop/Data/house_price.csv"
# ---------------------------------------------------------
csv_path = "/home/adarsh/Desktop/ml_l_class_ref/files/house_price_MLR.csv"

df = pd.read_csv(csv_path, delimiter=",")
print(df)
print("\n\n\n\n")



# -----------------------------------------
# Display statistics
print("\n\n\n\n")
print("DataFrame Statistics:")
print(df.describe())

print("end")
# -----------------------------------------





# -----------------------------------------
# Load AG News dataset (Huggingface)
# -----------------------------------------
dataset = load_dataset("ag_news")

df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

print("\n\n\n\n")
print(df_train.head())
print(df_test.head())
print("\n\n\n\n")





# ---------------------------------------------------------
# Creating new DataFrame and saving locally
# ---------------------------------------------------------
df_house_price = pd.DataFrame({
    'area':[2600,3000,3200,3600,4000],
    'rooms':[3,4,3,3,5],
    "age":[20,15,18,30,8],
    "price":[550000,565000,610000,595000,760000]
})
print(df_house_price)
print("\n\n\n\n")




# ---------------------------------------------------------
# SAVE LOCALLY — CHANGE PATH
# ---------------------------------------------------------
save_path = "/home/adarsh/Desktop/ml_l_class_ref/files/house_price_MLR.csv"
df_house_price.to_csv(save_path, index=False)

# ---------------------------------------------------------
# Read CSV without header
# ---------------------------------------------------------
df1 = pd.read_csv(save_path, delimiter=",", skiprows=1, header=None)
df1.columns = ["area", "rooms", "age", "price"]
print(df1)
print("\n\n\n\n")





# Read limited rows
df1 = pd.read_csv(save_path, delimiter=",", nrows=4)
print(df1)
print("\n\n\n\n")




# ---------------------------------------------------------
# Handling missing values in cricket dataset
# ---------------------------------------------------------
data = {
    "Name": ["Sachin", "Laxman", "Rahul", "Virat", "Rohit", "Kapil", "Sourav", "Bumrah"],
    "Age": [25, "n.a.", 35, "not available", 35, 65, 52, 30],
    "Salary": [50000, 60000, "n.a.", 55000, "n.a.", 72000, "not available", -1]
}
df_cricket = pd.DataFrame(data)
df_cricket.replace(["n.a.", "not available", -1], np.nan, inplace=True)
print(df_cricket)
print("\n\n\n\n")





# ---------------------------------------------------------
# Load weather data — CHANGE PATH
# ---------------------------------------------------------
weather_path = "/home/adarsh/Desktop/ml_l_class_ref/files/weather_data.csv"
df_weather = pd.read_csv(weather_path)
print(df_weather)
print("\n\n\n\n")






df_temp = df_weather.fillna(0)
df_new = df_weather.fillna({
    'temparature': 0,
    'windspeed': 0,
    'event': 'not known'
})

new_df = df_weather.ffill()  # forward fill
new_df1 = df_weather.bfill(axis="columns")
new_df = df_weather.ffill(limit=1)



# ---------------------------------------------------------
# Dictionary operations
# ---------------------------------------------------------
d1 = {"name": "David", "Age": 20, "place": "Delhi", "DOB": 1997}

# update
d1["name"] = "John"
d1.update({"name": "Manish"})

# add
d1["job"] = "teaching"

# remove
d1.pop("job")
d1.popitem()
del d1["Age"]

# clear and recreate
d1.clear()
d1 = {"name": "David", "Age": 20, "place": "Delhi", "DOB": 1997}

# copy
d2 = d1.copy()

# nested dictionary
d = {
    "student1":{"name":"XYZ","subject":"DA"},
    "student2":{"name":"PQR","subject":"CV"},
    "student3":{"name":"WXY","subject":"CC"}
}

print(d)
print("\n\n\n\n")
