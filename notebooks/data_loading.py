import pandas as pd

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

print("Fake News Dataset")
print(fake.head())

print("\nReal News Dataset")
print(real.head())

print("\nFake Dataset Shape:")
print(fake.shape)

print("\nReal Dataset Shape:")
print(real.shape)

print("\nFake Dataset Info:")
fake.info()

print("\nReal Dataset Info:")
real.info()

# Add labels
fake["label"] = 0
real["label"] = 1

print("\nFake Dataset with Label:")
print(fake.head())

print("\nReal Dataset with Label:")
print(real.head())

# Merge datasets
data = pd.concat([fake, real], ignore_index=True)

print("\nMerged Dataset Shape:")
print(data.shape)

print(data.head())

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

print("\nShuffled Dataset")
print(data.head())

# Save merged dataset
data.to_csv("dataset/news.csv", index=False)

print("\nnews.csv saved successfully!")