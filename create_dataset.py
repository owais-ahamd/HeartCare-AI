import pandas as pd
import os


url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv"


data = pd.read_csv(url)


os.makedirs("data", exist_ok=True)


data.to_csv(
    "data/heart.csv",
    index=False
)


print("Dataset saved successfully")
print(data.head())
print(data.shape)