import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), "heart.csv")

df = pd.read_csv(file_path)

print("Data Loaded Successfully!")
print(df.shape)