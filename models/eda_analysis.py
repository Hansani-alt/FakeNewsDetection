import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import Counter

# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("dataset/preprocessed_news.csv")

print("=" * 60)
print("              EDA - DATASET ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLabel Distribution:")
print(data["label"].value_counts())

fake_count = int((data["label"] == 0).sum())
real_count = int((data["label"] == 1).sum())

print("\nFAKE NEWS:", fake_count)
print("REAL NEWS:", real_count)

print("\nTotal Articles:", len(data))

# ============================================================
# CREATE SCREENSHOTS FOLDER
# ============================================================

os.makedirs("screenshots", exist_ok=True)

# ============================================================
# GRAPH 1 - CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

labels = ["FAKE NEWS", "REAL NEWS"]
values = [fake_count, real_count]

bars = plt.bar(labels, values)

plt.title("Fake News Dataset - Class Distribution")
plt.xlabel("News Class")
plt.ylabel("Number of Articles")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "screenshots/class_distribution.png",
    dpi=300
)

plt.close()

print("\nClass distribution graph saved.")


# ============================================================
# ARTICLE LENGTH ANALYSIS
# ============================================================

if "text" in data.columns:

    data["text_length"] = data["text"].astype(str).apply(
        lambda x: len(x.split())
    )

elif "tokens" in data.columns:

    data["text_length"] = data["tokens"].astype(str).apply(
        lambda x: len(x.split())
    )

else:

    data["text_length"] = 0


print("\nArticle Length Statistics:")
print(data["text_length"].describe())


# ============================================================
# GRAPH 2 - ARTICLE LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    data["text_length"],
    bins=50
)

plt.title("News Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    "screenshots/article_length_distribution.png",
    dpi=300
)

plt.close()

print("\nArticle length graph saved.")


# ============================================================
# MOST FREQUENT WORDS
# ============================================================

all_words = []

if "tokens" in data.columns:

    for value in data["tokens"].dropna():

        try:

            tokens = eval(value)

            if isinstance(tokens, list):

                all_words.extend(
                    [
                        str(word).lower()
                        for word in tokens
                        if str(word).isalpha()
                    ]
                )

        except:

            continue

elif "text" in data.columns:

    for text in data["text"].dropna():

        words = re.findall(
            r"\b[a-zA-Z]+\b",
            str(text).lower()
        )

        all_words.extend(words)


word_counts = Counter(all_words)

most_common = word_counts.most_common(20)

print("\nTop 20 Most Frequent Words:")

for word, count in most_common:

    print(
        f"{word}: {count}"
    )


# ============================================================
# GRAPH 3 - MOST FREQUENT WORDS
# ============================================================

words = [
    item[0]
    for item in most_common
]

counts = [
    item[1]
    for item in most_common
]

plt.figure(figsize=(10, 6))

plt.barh(
    words[::-1],
    counts[::-1]
)

plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()

plt.savefig(
    "screenshots/most_frequent_words.png",
    dpi=300
)

plt.close()

print("\nWord frequency graph saved.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("                 EDA COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print("1. screenshots/class_distribution.png")
print("2. screenshots/article_length_distribution.png")
print("3. screenshots/most_frequent_words.png")

print("\nEDA analysis completed successfully!")