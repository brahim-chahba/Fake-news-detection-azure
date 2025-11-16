import pandas as pd 
import os
befor_data="data/raw"
after_data="data\processed"

# we need to map the lables into integers
lable_map={
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

COLS = ["id",
    "label","statement","subject","speaker","speaker_job","state_info",
    "party","barely_true_counts","false_counts","half_true_counts",
    "mostly_true_counts","pants_on_fire_counts","context"
]

# if you ara in local uncomment this 
#def load_tsv_local(name_file):
  #path=os.path.join(befor_data,name_file)
  #df=pd_read_csv(path,sep="\t",header=None)
  #df.columns=COLS
  #return df
def load_tsv(name):
  df=pd.read_csv("train.tsv",sep="\t",header=None)
  df.columns=COLS
  return df

def clean(df):
  # only intrested in this 2
  df =df["label","statement"]
  # remove empty rows
  df=df.dropna()
  # remove duplicates in statments
  df=df.drop_duplicates(subset=["statement"])
  # remove spaces
  df["statement"]=df["statement"].str.strip()
  # map the labels
  df["label"]=df["label"].map(lable_map)
  # remove short statements 
  df =df[df["statement"].str.len()>10]
  return df
