import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier


data_dir = '../data/titanic/'

train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
num_train = len(train_data)

print(f'# of training samples: {num_train}')
print(f'# of test samples: {len(test_data)}')

all_features = pd.concat((train_data.iloc[:, 2:], test_data.iloc[:, 1:]))
print(f'features: {all_features.shape}')

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features['Embarked'] = all_features['Embarked'].fillna(all_features['Embarked'].mode().item())


# numeric_features_train = train_data.dtypes[train_data.dtypes != 'object'].index
# train_data[numeric_features_train] = train_data[numeric_features_train].fillna(0)

# numeric_features_test = test_data.dtypes[test_data.dtypes != 'object'].index
# test_data[numeric_features_test] = test_data[numeric_features_test].fillna(0)


# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)

# print(f"% of women who survived: {rate_women}")

# man = train_data.loc[train_data.Sex == 'male']['Survived']
# rate_man = sum(man) / len(man)

# print(f'% of man who survived: {rate_man}')

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", 'Embarked']
X = pd.get_dummies(all_features[:num_train][features])
X_test = pd.get_dummies(all_features[num_train:][features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_titanic.csv', index=False)
print("Your submission was successfully saved!")