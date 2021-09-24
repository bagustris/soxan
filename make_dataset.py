import os
import glob
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

data = []

for path in tqdm(Path("/home/aad13432ni/github/ser_greek/content/data/aesdd").glob("**/*.wav")):
    name = str(path).split('/')[-1].split('.')[0]
    label = str(path).split('/')[-2]
    
    try:
        # There are some broken files
        data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        # print(str(path), e)
        pass

    # break

df = pd.DataFrame(data)
df.head()

# Filter broken and non-existed paths

print(f"Step 0: {len(df)}")

df["status"] = df["path"].apply(
    lambda path: True if os.path.exists(path) else None)
df = df.dropna(subset=["path"])
df = df.drop("status", 1)
print(f"Step 1: {len(df)}")

df = df.sample(frac=1)
df = df.reset_index(drop=True)
df.head()

"""Let's explore how many labels (emotions) are in the dataset with what 
distribution."""

print("Labels: ", df["emotion"].unique())
print()
df.groupby("emotion").count()[["path"]]


"""For training purposes, we need to split data into train test sets; in this 
specific example, we break with a `20%` rate for the test set."""

save_path = "/home/aad13432ni/github/ser_greek/content/data/"

train_df, test_df = train_test_split(df, 
    test_size=0.2, random_state=101, stratify=df["emotion"])

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", 
                encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", 
               encoding="utf-8", index=False)