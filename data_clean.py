import pandas as pd
import os

base_dir = os.path.dirname(__file__)

file_path = os.path.join(
    base_dir,
    "house-price-dataset-of-india",
    "raw_data.csv"
)

df = pd.read_csv(file_path)

print("Loaded successfully:", df.shape)
print(df.head())

df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
print(df.columns)

df = df[[
    "price",
    "living_area",
    "number_of_bedrooms",
    "number_of_bathrooms",
    "number_of_floors",
    "condition_of_the_house",
    "grade_of_the_house",
    "built_year"
]]

#Feature Engineering
df["house_age"] = 2025 - df["built_year"]
df = df.drop("built_year", axis=1)

# ✅ SAVE CLEAN DATA HERE
df.to_csv("house-price-dataset-of-india/clean_data.csv", index=False)
print("Clean data saved! Shape:", df.shape)