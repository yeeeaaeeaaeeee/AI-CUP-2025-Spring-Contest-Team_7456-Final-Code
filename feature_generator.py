import numpy as np
import pandas as pd
import csv
from pathlib import Path

def standardized_moment_calc(data, power_factor, axis=None, eps=1e-8):
    err = data - np.mean(data, axis=axis)
    devi = np.sqrt(np.var(data, axis=axis))
    devi = np.where(devi == 0, eps, devi)  # Avoid division by zero
    return np.mean(np.power(err / devi, power_factor), axis=axis)

def feature(data,mode: int):

    concat_list = []

    mean = np.mean(data,axis=0)
    var = np.var(data,axis=0)
    rms = np.sqrt(np.mean(np.pow(data,2),axis=0))
    mx = np.max(data,axis=0)
    mn = np.min(data,axis=0)
    range = mx - mn
    argmx_ratio = np.argmax(data,axis=0)/data.shape[0]
    skew = standardized_moment_calc(data,3,axis=0)
    kurt = standardized_moment_calc(data,4,axis=0)
    concat_list.extend([mean,var,rms,mx,mn,range,argmx_ratio,skew,kurt])

    acc = data[:,:3]
    acc = np.linalg.norm(acc,axis=1)
    amean = np.mean(acc)
    avar = np.var(acc)
    arms = np.sqrt(np.mean(np.pow(acc,2)))
    amx = np.max(acc)
    amn = np.min(acc)
    arange = amx - amn
    aargmx_ratio = np.argmax(acc)/acc.shape[0]
    askew = standardized_moment_calc(acc,3)
    akurt = standardized_moment_calc(acc,4)
    concat_list.extend([[amean],[avar],[arms],[amx],[amn],[arange],[aargmx_ratio],[askew],[akurt]])
    
    gro = data[:,3:]
    gro = np.linalg.norm(gro,axis=1)
    gmean = np.mean(gro)
    gvar = np.var(gro)
    grms = np.sqrt(np.mean(np.pow(gro,2)))
    gmx = np.max(gro)
    gmn = np.min(gro)
    grange = gmx - gmn
    gargmx_ratio = np.argmax(gro)/gro.shape[0]
    gskew = standardized_moment_calc(gro,3)
    gkurt = standardized_moment_calc(gro,4)
    concat_list.extend([[gmean],[gvar],[grms],[gmx],[gmn],[grange],[gargmx_ratio],[gskew],[gkurt]])
    
    return np.concat(concat_list)

def add_gaussian_noise(data, std_ratio=0.03):
    std = np.std(data, axis=0)
    noise = np.random.normal(0, std_ratio * std, data.shape)
    return data + noise

def feature_generate(data_path, info_path, cut_points_path, feature_path, headerList, jerk=False, add_noise=False, noise_fn=add_gaussian_noise):
    cnt = [[] for i in range(100)]
    pathlist_txt = list(Path(data_path).glob('**/*.txt'))
    pathlist_txt.sort(key=lambda x: int(str(x).split('\\')[-1][:-4]))
    info = pd.read_csv(info_path)
    train_cut_points = pd.read_csv(cut_points_path)

    All_outputs = [[] for _ in range(len(pathlist_txt))]
    szd = dict({i: 0 for i in range(30)})

    for file_id, file in enumerate(pathlist_txt):
        unique_id = int(Path(file).stem)
        All_data = []

        count = 0
        with open(file) as f:
            for line in f.readlines():
                if line == '\n' or count == 0:
                    count += 1
                    continue
                num = list(map(int, line.split(' ')))
                All_data.append(num)

        cut_points = train_cut_points[train_cut_points['unique_id'] == unique_id]['cut_points']
        cut_points = cut_points.values[0].replace('\n', '').replace('[', '').replace(']', '')
        cut_points = np.array(list(map(int, cut_points.split())))
        cnt[len(cut_points)].append(unique_id)

        All_data = np.array(All_data)

        for cut_id in range(len(cut_points) - 1):
            segment = All_data[cut_points[cut_id]: cut_points[cut_id + 1]]
            if add_noise:
                segment = noise_fn(segment)

            
            All_outputs[file_id].append(feature(segment, None))

        all_mean = np.mean(All_outputs[file_id], axis=0)
        all_var = np.var(All_outputs[file_id], axis=0)
        all_rms = np.sqrt(np.mean(np.power(All_outputs[file_id], 2), axis=0))
        all_max = np.max(All_outputs[file_id], axis=0)
        all_min = np.min(All_outputs[file_id], axis=0)
        all_range = all_max - all_min
        all_argmax = np.argmax(All_outputs[file_id], axis=0)
        all_argmin = np.argmin(All_outputs[file_id], axis=0)
        all_arg_range = all_argmax - all_argmin
        all_skew = standardized_moment_calc(All_outputs[file_id], 3, axis=0)
        all_kurt = standardized_moment_calc(All_outputs[file_id], 4, axis=0)

        All_outputs[file_id].extend([all_mean, all_var, all_rms, all_max, all_min,
                                     all_range, all_argmax, all_argmin, all_arg_range, all_skew, all_kurt])

    for file_id, file in enumerate(pathlist_txt):
        unique_id = int(Path(file).stem)
        with open(f'{feature_path}/{unique_id}.csv', 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            for row in All_outputs[file_id]:
                writer.writerow(np.array(row, dtype=float))

    print("Feature generation done!")
    return cnt