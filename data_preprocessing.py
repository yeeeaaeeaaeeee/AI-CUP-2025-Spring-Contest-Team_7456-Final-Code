import numpy as np
import pandas as pd
from pathlib import Path

def preprocess(data, file_id, row, min_length, max_length, extra_length, block_length, test=False, cut_trail=False, cut_front=False):
    if not test and len(data) < min_length:
        print(f"Length of {file_id} is less than {min_length}")
        return None
    block_count = (data.shape[0]-extra_length) // block_length
    if cut_trail:
        extra_data = data[-extra_length:]
        front_data = data[:-extra_length]
        if len(front_data)+len(extra_data) > max_length:
            front_data = front_data[block_length:]
        if len(front_data)+len(extra_data) > max_length:
            front_data = front_data[:-block_length]
        data = np.concatenate([front_data, extra_data])
    if len(data) > max_length:
        if cut_front:
            extra_data = data[-extra_length:]
            front_data = data[:-extra_length]
            front_data = front_data[:max_length-len(extra_data)]
            data = np.concatenate([front_data, extra_data])
        else:
            data = data[-max_length:]
    if np.isinf(data).sum() > 0:
        print(f"NaN in {file_id}")
    if np.isnan(data).sum() > 0:
        print(f"NaN in {file_id}")
    data = np.pad(data, (max_length - len(data), 0), 'constant', constant_values=np.nan)
    mode_array = np.zeros((10,))
    mode_array[int(row['mode'].values[0])-1] = 1
    data = np.concatenate((data, mode_array))
    return data

def read_feature(info, path, min_length, max_length, extra_length, block_length, test=False, cut_trail=False, cut_front=False , feature_mask = None):

    # 讀取特徵 CSV 檔（位於 "data/tabular_data_train"）
    datalist = list(Path(path).glob('**/*.csv'))
    datalist.sort(key=lambda x: int(x.stem))
    # print(datalist)
    target_mask = ['gender', 'hold racket handed', 'play years', 'level', 'player_id']

    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    total_discard_count = 0

    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue

        data = pd.read_csv(file)
        if feature_mask is not None:
            data = data[feature_mask]
        
        if not test:
            target = row[target_mask].copy()  # Create a copy to avoid SettingWithCopyWarning
            
            # Transform gender: 0->2, 1->1
            target['gender'] = 2 - target['gender']
            target['hold racket handed'] = 2 - target['hold racket handed']

        #flatten data and pad it to length max_length
        data = data.values.flatten()
        # print(data)
        data = preprocess(data, unique_id, row, min_length, max_length, extra_length, block_length, test=test, cut_trail=cut_trail, cut_front=cut_front)
        if data is None:
            total_discard_count+=1
            continue
        data_df = pd.DataFrame([data])
        x_train = pd.concat([x_train, data_df], ignore_index=True)
        if not test:
            y_train = pd.concat([y_train, target], ignore_index=True)
        # print(x_train.shape)
    print(f"Total discarded samples: {total_discard_count}")
    return x_train, y_train