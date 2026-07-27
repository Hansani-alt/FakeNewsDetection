import pandas as pd


def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    return df


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()

    return df


if __name__ == "__main__":
    fake_path = "dataset/Fake.csv"
    true_path = "dataset/True.csv"

    data = load_data(fake_path, true_path)
    data = clean_data(data)

    data.to_csv("dataset/clean_news.csv", index=False)

    print("Data cleaned successfully.")
    print(data.head())
