import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv("data/ml100k.csv")
print(len(data.user.unique()),len(data.item.unique()))
# specify the columns to balance the split on
column1 = 'user'
column2 = 'item'

# create a 2D array containing the features to stratify on
stratify_on = data[[column1, column2]]
counts = data[[column1, column2]].value_counts()
print(counts)
exit()

# create the StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

# split the data into training and test sets, with balanced ratios for both columns
for train_index, test_index in sss.split(data, stratify_on):
    train_data = data.loc[train_index]
    test_data = data.loc[test_index]

# check the ratios of values in the balanced columns in both sets
print(len(train_data.user.unique()),len(train_data.item.unique()))
print(len(test_data.user.unique()),len(test_data.item.unique()))