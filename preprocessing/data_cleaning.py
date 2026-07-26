import pandas as pd
import re
import string


# Load dataset
data = pd.read_csv("dataset/news.csv")


# 1. Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# 2. Check duplicate rows
print("\nDuplicate Rows:")
print(data.duplicated().sum())


# 3. Remove duplicate rows
data = data.drop_duplicates()

print("\nShape after removing duplicates:")
print(data.shape)


# 4. Convert text to lowercase
data["text"] = data["text"].str.lower()

print("\nLowercase Conversion Completed!")
print(data["text"].head())


# 5. Remove URLs
data["text"] = data["text"].apply(
    lambda x: re.sub(r"http\S+|www\S+", "", x)
)


# 6. Remove HTML tags
data["text"] = data["text"].apply(
    lambda x: re.sub(r"<.*?>", "", x)
)


# 7. Remove punctuation
data["text"] = data["text"].apply(
    lambda x: x.translate(str.maketrans("", "", string.punctuation))
)


# 8. Remove numbers
data["text"] = data["text"].apply(
    lambda x: re.sub(r"\d+", "", x)
)


# 9. Remove extra spaces
data["text"] = data["text"].apply(
    lambda x: " ".join(x.split())
)


# Save cleaned dataset
data.to_csv("dataset/clean_news.csv", index=False)


print("\nClean dataset saved successfully!")
