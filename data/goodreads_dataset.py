import pandas as pd
import gdown
import os
from sklearn.model_selection import train_test_split
import logging

DATA_DIR = "."

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

datasets = {
    "goodreads_reviews_spoiler.json.gz": "196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS",
    "goodreads_book_genres_initial.json.gz": "1ah0_KpUterVi-AHxJ03iKD6O0NfbK0md",
}

print("Getting data...")
for name, id in datasets.items():
    if not os.path.exists(os.path.join(DATA_DIR, name)):
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, output=DATA_DIR)


print("Reading data...")
review_df = pd.read_json(
    os.path.join(DATA_DIR, "goodreads_reviews_spoiler.json.gz"), lines=True,
)
genres_df = pd.read_json(
    os.path.join(DATA_DIR, "goodreads_book_genres_initial.json.gz"), lines=True
)

print("Merging data...")
full_df = pd.merge(review_df, genres_df, on="book_id")


def save_dataset_splits(df, name):
    train_and_val, test = train_test_split(
        df, test_size=0.2, random_state=44, stratify=df["has_spoiler"]
    )

    train, val = train_test_split(
        train_and_val,
        test_size=0.25,
        random_state=44,
        stratify=train_and_val["has_spoiler"],
    )

    train.to_json(
        os.path.join(DATA_DIR, f"{name}-train.json.gz"), orient="records", lines=True
    )
    val.to_json(
        os.path.join(DATA_DIR, f"{name}-val.json.gz"), orient="records", lines=True
    )
    test.to_json(
        os.path.join(DATA_DIR, f"{name}-test.json.gz"), orient="records", lines=True
    )


print("Splitting data...")
save_dataset_splits(full_df, "goodreads")

spoiler_df = full_df[full_df["has_spoiler"] == True]
nonspoiler_df = full_df[full_df["has_spoiler"] == False]
nonspoiler_df_subset = nonspoiler_df.sample(len(spoiler_df), random_state=44)
balanced_df = pd.concat([spoiler_df, nonspoiler_df_subset])
save_dataset_splits(balanced_df, "goodreads_balanced")
