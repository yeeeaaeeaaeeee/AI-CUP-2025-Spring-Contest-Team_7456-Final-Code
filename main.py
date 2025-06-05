import data_preprocessing as dp
import feature_generator as fg
import model_training_and_inferencing as mdti
import numpy as np
import pandas as pd
import random
from pathlib import Path
import csv
from datetime import datetime as dt
import os

train_path = 'data/39_Training_Dataset/'
test_path = 'data/39_Test_Dataset/'
train_feature_path = 'data/train_feature/'
test_feature_path = 'data/test_feature/'
os.makedirs(train_feature_path, exist_ok=True)
os.makedirs(test_feature_path, exist_ok=True)
RANDOM_SEED = 42111
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
headerList = [
                'ax_mean','ay_mean','az_mean','gx_mean','gy_mean','gz_mean',
                'ax_var','ay_var','az_var','gx_var','gy_var','gz_var',
                'ax_rms','ay_rms','az_rms','gx_rms','gy_rms','gz_rms',
                'ax_max','ay_max','az_max','gx_max','gy_max','gz_max',
                'ax_min','ay_min','az_min','gx_min','gy_min','gz_min',
                'ax_range','ay_range','az_range','gx_range','gy_range','gz_range',
                'ax_argmax_ratio','ay_argmax_ratio','az_argmax_ratio','gx_argmax_ratio','gy_argmax_ratio','gz_argmax_ratio',
                'ax_skew','ay_skew','az_skew','gx_skew','gy_skew','gz_skew',
                'ax_kurt','ay_kurt','az_kurt','gx_kurt','gy_kurt','gz_kurt',
                'a_mean','a_var','a_rms','a_max','a_min','a_range','a_argmax_ratio','a_skew','a_kurt',
                'g_mean','g_var','g_rms','g_max','g_min','g_range','g_argmax_ratio','g_skew','g_kurt',
            ]
extra = ['mean','var','rms','max','min','range','argmax','argmin','arg_range','skew','kurt']

min_length = len(headerList)*(23+len(extra))
max_length = len(headerList)*(29+len(extra))
block_length = len(headerList)
extra_length = len(headerList)*len(extra)
print(f'max_length: {max_length}')

general_model_setting = {
    "random_state" : RANDOM_SEED,
}

# Generate features
fg.feature_generate(train_path+'train_data/',train_path+'train_info.csv',train_path+'train_cut_points.csv',train_feature_path, headerList)
fg.feature_generate(test_path+'test_data/',test_path+'test_info.csv',test_path+'test_cut_points.csv',test_feature_path, headerList)

info = pd.read_csv(train_path + 'train_info.csv')
x_train, y_train = dp.read_feature(info, train_feature_path, min_length, max_length, extra_length, block_length, test=False, cut_trail=False)

# Imports
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Gender Train
gender_model_setting = general_model_setting.copy()

X_train_gender = x_train.copy()

le = LabelEncoder()
y_train_le_gender = le.fit_transform(y_train['gender'])

neg = (y_train_le_gender == 0).sum()
pos = (y_train_le_gender == 1).sum()
scale = neg / pos

print(f"Gender scale_pos_weight: {scale}")

gender_model_setting['scale_pos_weight'] = scale # Default
gender_mask = mdti.KFoldById(X_train_gender, y_train_le_gender, y_train['player_id'], gender_model_setting, RANDOM_SEED, K=4, model_type=XGBClassifier, top_n_features=[feature_count for feature_count in range(500,3000,500)])
_, _, gender_preds_train = mdti.KFoldEnsembleByMasks(X_train_gender, y_train_le_gender, y_train['player_id'], gender_model_setting, RANDOM_SEED, gender_mask, K=4, model_type=XGBClassifier)

# Hold Racket Handed Train
hold_model_setting = general_model_setting.copy()

X_train_hold = x_train.copy()

le = LabelEncoder()
y_train_le_hold = le.fit_transform(y_train['hold racket handed'])

neg = (y_train_le_hold == 0).sum()
pos = (y_train_le_hold == 1).sum()
scale = neg / pos
print(f"hold racket handed scale_pos_weight: {scale}")

hold_model_setting['scale_pos_weight'] = scale
hold_mask = mdti.KFoldById(X_train_hold, y_train_le_hold, y_train['player_id'], hold_model_setting, RANDOM_SEED, K=4, model_type=XGBClassifier, top_n_features=[feature_count for feature_count in range(500,3000,500)])

# Levels Train

level_model_setting = general_model_setting.copy()

X_train_level = x_train.copy()

le = LabelEncoder()
y_train_le_level = le.fit_transform(y_train['level'])

gender_pred_train = gender_preds_train.copy()
# gender_pred_train_df = pd.DataFrame({'gender_pred': gender_pred_train.values}, index=X_train_level.index)
gender_pred_train_df = pd.DataFrame(gender_pred_train, columns=['gender_pred'], index=X_train_level.index)
gender_pred_train_df = gender_pred_train_df.apply(pd.to_numeric)
X_train_level = pd.concat([X_train_level,gender_pred_train_df], axis=1)

level_mask = mdti.KFoldById(X_train_level, y_train_le_level, y_train['player_id'], level_model_setting, RANDOM_SEED, K=2, model_type=XGBClassifier, multi=True, top_n_features=[feature_count for feature_count in range(500,3000,500)])

_, _, level_preds_train = mdti.KFoldEnsembleByMasks(X_train_level, y_train_le_level, y_train['player_id'], level_model_setting, RANDOM_SEED, level_mask, K=2, model_type=XGBClassifier,multi=True)

# Years Train

years_model_setting = general_model_setting.copy()

X_train_years = x_train.copy()

le = LabelEncoder()
y_train_le_years = le.fit_transform(y_train['play years'])

gender_pred_train = y_train['gender'].copy()
gender_pred_train_df = pd.DataFrame({'gender_pred': gender_pred_train.values}, index=X_train_level.index)
# gender_pred_train_df = pd.DataFrame(gender_pred_train, columns=['gender_pred'], index=X_train_years.index)
gender_pred_train_df = gender_pred_train_df.apply(pd.to_numeric)

level_pred_train = level_preds_train.copy()
# level_pred_train_onehot = pd.get_dummies(level_pred_train, prefix='level_pred')
# level_pred_train_df = level_pred_train_onehot.set_index(X_train_years.index)
level_pred_train_df = pd.DataFrame(level_pred_train, columns=['level_pred_2','level_pred_3','level_pred_4','level_pred_5'], index=X_train_years.index)
level_pred_train_df = level_pred_train_df.apply(pd.to_numeric)
level_pred_train_df = level_pred_train_df.apply(pd.to_numeric)

X_train_years = pd.concat([X_train_years,gender_pred_train_df,level_pred_train_df], axis=1)

years_mask = mdti.KFoldById(X_train_years, y_train_le_years, y_train['player_id'], years_model_setting, RANDOM_SEED, K=3, model_type=XGBClassifier, multi=True, top_n_features=[feature_count for feature_count in range(500,3000,500)])

# Load test data
info = pd.read_csv(test_path+'test_info.csv')
x_test, _ = dp.read_feature(info, test_feature_path, min_length, max_length, extra_length, block_length, test=True, cut_trail=False)

# Gender Test
X_train_gender = x_train.copy()
X_test_gender = x_test.copy()

le = LabelEncoder()
y_train_le_gender = le.fit_transform(y_train['gender'])
gender_model_setting = general_model_setting.copy()
# Calculate scale_pos_weight
neg = (y_train_le_gender == 0).sum()
pos = (y_train_le_gender == 1).sum()
scale = neg / pos
print(f"scale_pos_weight: {scale}")
gender_model_setting['scale_pos_weight'] = scale

gender_pred_test, gender_pred_train = mdti.get_test_ensemble_results(X_train_gender, y_train_le_gender, X_test_gender, gender_model_setting, gender_mask, multi=False, model_type=XGBClassifier)

# Hold Racket Handed Test
X_train_hold = x_train.copy()
X_test_hold = x_test.copy()

le = LabelEncoder()
y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
hold_model_setting = general_model_setting.copy()
# Calculate scale_pos_weight
neg = (y_train_le_hold == 0).sum()
pos = (y_train_le_hold == 1).sum()
scale = neg / pos

print(f"scale_pos_weight: {scale}")

hold_pred_test, hold_pred_train = mdti.get_test_ensemble_results(X_train_hold, y_train_le_hold, X_test_hold, hold_model_setting, hold_mask, multi=False, model_type=XGBClassifier)

# Levels Test

X_train_level = x_train.copy()
X_test_level = x_test.copy()

le = LabelEncoder()
y_train_le_level = le.fit_transform(y_train['level'])
level_model_setting = general_model_setting.copy()

gender_pred_train = y_train['gender'].copy()
gender_pred_train_df = pd.DataFrame({'gender_pred': gender_pred_train.values}, index=X_train_level.index)
gender_pred_train_df = gender_pred_train_df.apply(pd.to_numeric)

gender_pred_test_df = pd.DataFrame(gender_pred_test, columns=['gender_pred'], index=X_test_level.index)
gender_pred_test_df = gender_pred_test_df.apply(pd.to_numeric)

X_train_level = pd.concat([X_train_level,gender_pred_train_df], axis=1)
X_test_level = pd.concat([X_test_level,gender_pred_test_df], axis=1)

level_pred_test, level_pred_train = mdti.get_test_ensemble_results(X_train_level, y_train_le_level, X_test_level, level_model_setting, level_mask, model_type=XGBClassifier, multi=True)

# Years Test

X_train_years = x_train.copy()
X_test_years = x_test.copy()

le = LabelEncoder()
y_train_le_years = le.fit_transform(y_train['play years'])
years_model_setting = general_model_setting.copy()

gender_pred_train = y_train['gender']
gender_pred_train_df = pd.DataFrame({'gender_pred': gender_pred_train.values}, index=X_train_years.index)
gender_pred_train_df = gender_pred_train_df.apply(pd.to_numeric)

gender_pred_test_df = pd.DataFrame(gender_pred_test, columns=['gender_pred'], index=X_test_years.index)
gender_pred_test_df = gender_pred_test_df.apply(pd.to_numeric)

level_pred_train_df = pd.DataFrame(level_pred_train, columns=['level_pred_2','level_pred_3','level_pred_4','level_pred_5'], index=X_train_years.index)
level_pred_train_df = level_pred_train_df.apply(pd.to_numeric)

level_pred_test_df = pd.DataFrame(level_pred_test, columns=['level_pred_2','level_pred_3','level_pred_4','level_pred_5'], index=X_test_years.index)
level_pred_test_df = level_pred_test_df.apply(pd.to_numeric)

X_train_years = pd.concat([X_train_years,gender_pred_train_df,level_pred_train_df], axis=1)
X_test_years = pd.concat([X_test_years,gender_pred_test_df,level_pred_test_df], axis=1)

years_pred_test, years_pred_train = mdti.get_test_ensemble_results(X_train_years, y_train_le_years, X_test_years, years_model_setting, years_mask, model_type=XGBClassifier, multi=True)

# Output results
os.makedirs('submissions', exist_ok=True)
submission_path = f"submissions/submission-{dt.today().strftime('%Y-%m-%d_%H-%M-%S')}-RFC.csv"
HeaderList = ['unique_id',
              'gender','hold racket handed',
              'play years_0','play years_1','play years_2',
              'level_2','level_3','level_4','level_5']

test_datalist = list(Path(test_feature_path).glob('**/*.csv'))
test_datalist.sort(key=lambda x: int(x.stem))
unique_ids = np.array([[int(Path(file).stem)] for file in test_datalist])

output = np.concat([unique_ids,gender_pred_test[:,None],hold_pred_test[:,None],years_pred_test,level_pred_test],axis=1)
output = np.round(output,decimals=4)

print(f'Submission shape: {output.shape}')

with open(submission_path, 'w+', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(HeaderList)
    for row in output:
        row = list(row)
        row[0] = int(row[0])
        writer.writerow(row)