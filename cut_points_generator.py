import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    # Read data from file
    data = []
    count = 0
    with open(file_path) as f:
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = list(map(int, line.split(' ')))
            data.append(num)
    
    return np.array(data)

def moving_average(arr, window_size):
    """
    Calculate the average of each element with its neighbors within a specified window size.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        1D input array
    window_size : int
        The number of elements to include on each side of the center element
        
    Returns:
    --------
    numpy.ndarray
        Array of same size as input with each element replaced by the average of
        itself and its neighbors within window_size
    """
    result = np.zeros_like(arr, dtype=float)
    n = len(arr)
    
    for i in range(n):
        # Calculate the window boundaries
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)
        
        # Calculate average for current position
        result[i] = np.mean(arr[left:right])
    
    return result

def visualize_sensor_data(data,window_size=5, cutpoints=None, title=None,axis=None):
    """
    Visualize accelerometer and gyroscope data from a file.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The data file
    title : str, optional
        Plot title
    """
    
    # Create time axis
    time = np.arange(len(data))
    
    # Create subplots for better visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    acc = data[:, 0:3]
    acc = np.pow(acc, 2)
    acc = np.sum(acc, axis=1)
    acc = np.sqrt(acc)
    gyro = data[:, 3:6]
    gyro = np.pow(gyro, 2)
    gyro = np.sum(gyro, axis=1)
    gyro = np.sqrt(gyro)

    # Plot accelerometer data
    ax1.plot(time, moving_average(data[:, 0],window_size), 'r-', label='ax')
    ax1.plot(time, moving_average(data[:, 1],window_size), 'g-', label='ay')
    ax1.plot(time, moving_average(data[:, 2],window_size), 'b-', label='az')
    ax1.plot(time, moving_average(acc,window_size), 'k-', label='a')
    
    if cutpoints is not None:
        for cp in cutpoints:
            ax1.axvline(x=cp, color='purple', linestyle='--', alpha=0.5)
            ax1.plot(cp, moving_average(acc, window_size)[cp], 'ro', markersize=8)
    
    ax1.set_ylabel('Acceleration')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title('Accelerometer Data')

    
    
    # Plot gyroscope data
    ax2.plot(time, moving_average(data[:, 3],window_size), 'c-', label='gx')
    ax2.plot(time, moving_average(data[:, 4],window_size), 'm-', label='gy')
    ax2.plot(time, moving_average(data[:, 5],window_size), 'y-', label='gz')
    ax2.plot(time, moving_average(gyro,window_size), 'k-', label='g')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular Velocity')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Gyroscope Data')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    return 

def detect_local_minimum(data,window_size):
    """
    Detect local minimums in the data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        1D input array
    window_size : int
        The number of elements to include on each side of the center element
        
    Returns:
    --------
    list
        Indices of local minimums in the data
    """
    local_minimums = []
    
    for i in range(len(data)):
        # Calculate the window boundaries
        left = max(0, i - window_size)
        right = min(len(data), i + window_size + 1)
        
        # Check if current element is a local minimum
        if data[i] == min(data[left:right]):
            local_minimums.append(i)
    
    return local_minimums

def get_cut_points(data,window_size=50):
    """
    Detect cut points in the data based on local minimums.
    
    Parameters:
    data : numpy.ndarray
        Input data array.
    window_size : int, optional
        Size of the window for detecting local minimums (default is 50).
    """
    print(data.shape)
    acc = np.pow(data[:, 0:3], 2)
    acc = np.sum(acc, axis=1)
    acc = np.sqrt(acc)
    lm = detect_local_minimum(acc, window_size)
    print(len(lm),lm)
    return lm

def get_cut_points_new(acc,window_size=50):
    """
    Detect cut points in the data based on local minimums.
    
    Parameters:
    data : numpy.ndarray
        Input data array.
    """
    lm = detect_local_minimum(acc, window_size)
    mx = np.max(acc)
    if lm[0]!=0 and np.max(acc[0:lm[0]])-acc[lm[0]]>0.5*mx and abs(acc[0]-acc[lm[0]])<0.3*(np.max(acc[0:lm[0]])-acc[lm[0]]):
        lm.insert(0,0)
    
    if lm[-1]!=len(acc)-1 and np.max(acc[lm[-1]:-1])-acc[lm[-1]]>0.5*mx and abs(acc[-1]-acc[lm[-1]])<0.3*(np.max(acc[lm[-1]:-1])-acc[lm[-1]]):
        lm.append(len(acc)-1)
    
    # print(len(lm),lm)
    return lm
    
def AMPD(data):
    """
    实现AMPD算法
    :param data: 1-D numpy.ndarray 
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]

def cut(x,threshold=None,backtrace=10):#輸入絕對值陣列 回傳砍過的陣列 前面砍的項數 和後面砍的項數
    if threshold == None:
        threshold = np.max(x)*0.3
    mx=x[0]
    mn=x[0]
    flag1=0
    flag2=0
    for i in range(len(x)):
        if x[i]>mx:
            mx=x[i]
        if x[i]<mn:
            mn=x[i]
        if mx-mn>threshold:
            flag1=i
            break
    mx=x[-1]
    mn=x[-1]
    for i in range(len(x)):
        if x[-i-1]>mx:
            mx=x[-i-1]
        if x[-i-1]<mn:
            mn=x[-i-1]
        if mx-mn>threshold:
            flag2=i
            break
    
    # flag 1~n-1
    cut1 = max(0,flag1-1-backtrace)
    cut2 = min(-1,-flag2+backtrace)
    return x[cut1:cut2],cut1,-cut2 #回傳砍過的陣列 前面砍的項數 和後面砍的項數

def get_cut_points_from_file(path,data_id=None, save_pic=False):
    """
    Detect cut points in the data based on local minimums.
    """

    data = load_data(path)
    # 例如只用加速度大小
    x = np.sqrt(np.sum(np.power(data[:, 0:3], 2), axis=1))
    # 在這裡加入去頭去尾
    x, x1, x2 = cut(x)
    # peaks, _ = scipy.signal.find_peaks(x, height=4000, distance=50)
    T = AMPD(x)
    peaks = get_cut_points_new(x, window_size=int((T[-1] - T[0])/(len(T)-1)/2))
    # 計算相鄰峰值的間隔
    intervals = np.diff(peaks)

    # 判斷分布是否均勻（以標準差為指標）
    std_interval = np.std(intervals)
    mean_interval = np.mean(intervals)
    if std_interval > 0.2 * mean_interval:
        print("峰值分布不均勻")
    if save_pic:
        # 重新載入原始資料長度
        x_full = np.sqrt(np.sum(np.power(data[:, 0:3], 2), axis=1))
        plt.figure(figsize=(12, 6))
        plt.plot(x_full, label='raw_data')
        # 標示被 cut 掉的頭尾
        plt.axvspan(0, x1, color='red', alpha=0.3, label=f'head cut: {x1} point')
        plt.axvspan(len(x_full)-x2, len(x_full), color='orange', alpha=0.3, label=f'tail cut: {x2} point')
        # 標示 cut 後的資料
        plt.plot(np.arange(x1, len(x_full)-x2), x, label='cut_data')
        # 標示峰值
        peaks_arr = np.array(peaks)  # 新增這行
        plt.plot(peaks_arr + x1, x[peaks_arr], "o", label='peaks')
        plt.title(f"scipy.signal.find_peaks, id={data_id}")
        plt.legend()
        plt.savefig(f"plot/id={data_id}.png")
    plt.close()

    return np.array(peaks)+x1

if __name__=='__main__':

    corrupted_ids = [2025, 2179]
    ###### Corrupted data : 2025, 2179 #######
    train_info = pd.read_csv("data/39_Training_Dataset/train_info.csv")

    unique_ids = []
    all_cut_points = []

    for i in train_info['unique_id']:
        if i in corrupted_ids:
            unique_ids.append(i)
            all_cut_points.append(np.array([0,load_data(f"data/39_Training_Dataset/train_data/{i}.txt").shape[0]-1]))
            continue
        peaks = get_cut_points_from_file(f"data/39_Training_Dataset/train_data/{i}.txt", save_pic=False)
        
        # Append data to lists
        unique_ids.append(i)
        all_cut_points.append(peaks)
        
        print(f"ID {i} processed.")

    # Create the DataFrame
    train_cut_points = pd.DataFrame({
        'unique_id': unique_ids,
        'cut_points': all_cut_points
    })

    # Save the DataFrame to CSV
    train_cut_points.to_csv("data/39_Training_Dataset/train_cut_points.csv", index=False)

    test_info = pd.read_csv("data/39_Test_Dataset/test_info.csv")

    unique_ids = []
    all_cut_points = []

    for i in test_info['unique_id']:
        if i in corrupted_ids:
            unique_ids.append(i)
            all_cut_points.append(np.array([0,load_data(f"data/39_Test_Dataset/test_data/{i}.txt").shape[0]-1]))
            continue
        peaks = get_cut_points_from_file(f"data/39_Test_Dataset/test_data/{i}.txt", save_pic=False)
        
        # Append data to lists
        unique_ids.append(i)
        all_cut_points.append(peaks)
        
        print(f"ID {i} processed.")

    # Create the DataFrame
    test_cut_points = pd.DataFrame({
        'unique_id': unique_ids,
        'cut_points': all_cut_points
    })

    # Save the DataFrame to CSV
    test_cut_points.to_csv("data/39_Test_Dataset/test_cut_points.csv", index=False)