import pandas as pd
import os

# use this if you are using colab 
before_data = "/content/data/raw"
after_data  = "/content/data/processed"

# label mapping
label_map = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

# column names
COLS = [
    "id","label","statement","subject","speaker","speaker_job","state_info",
    "party","barely_true_counts","false_counts","half_true_counts",
    "mostly_true_counts","pants_on_fire_counts","context"
]

# Load TSV file
def load_tsv(name_file):
    path = os.path.join(before_data, name_file)
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = COLS
    return df

# Clean dataframe
def clean(df):
    # we are intrested in only two columns
    df = df[["label", "statement"]]

    df = df.dropna()                                  # remove empty rows
    df = df.drop_duplicates(subset=["statement"])      # remove duplicate texts
    df["statement"] = df["statement"].str.strip()      # remove whitespace
    df["label"] = df["label"].map(label_map)           # convert label strings → ints

    # keep only statements longer than 10 chars
    df = df[df["statement"].str.len() > 10]

    return df

def main():
    # create output folder id it does not exist 
    os.makedirs(after_data, exist_ok=True)

    # correct file names
    splits = ["train.tsv", "test.tsv", "valid.tsv"]

    for split in splits:
        print(f"Cleaning... {split}")

        df = load_tsv(split)
        df = clean(df)

        save_name = split.replace(".tsv", "_clean.tsv")
        df.to_csv(os.path.join(after_data, save_name), index=False)

        print(f"Cleaned {split}")
        print(df.head())
        print(df["label"].value_counts())
        print("-" * 40)

# correct main check
if __name__ == "__main__":
    main()
