import pandas as pd
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader
import sys

if len(sys.argv)<2:
    print("Usage: python split.py <datafile>")
    exit()
if len(sys.argv)==3:
    max_rating = int(sys.argv[2])
else:
    max_rating = 5

df = pd.read_csv(sys.argv[1])
reader = Reader(rating_scale=(1, max_rating))
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.2, random_state=42)

train_data = {'user':[],'item':[],'rating':[]}
for uid,iid,rating in trainset.all_ratings():
    train_data["rating"].append(rating)
    train_data["user"].append(trainset.to_raw_uid(uid))
    train_data["item"].append(trainset.to_raw_iid(iid))

test_data = {'user':[],'item':[],'rating':[]}
for uid,iid,rating in testset:
    test_data["rating"].append(rating)
    test_data["user"].append(uid)
    test_data["item"].append(iid)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
print(len(train_data.user.unique()),len(train_data.item.unique()))
print(len(test_data.user.unique()),len(test_data.item.unique()))
print(set(test_data.user).difference(set(train_data.user)))
print(set(test_data.item).difference(set(train_data.item)))

train_data.to_csv("train.csv",index=False)
test_data.to_csv("test.csv",index=False)