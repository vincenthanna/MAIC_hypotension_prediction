import numpy as np
from scipy.signal import find_peaks

sample_rate = 100

def moving_average(a, n=200):
    '''
    각각 size=n 만큼의 윈도우의 평균값을 갖게한다.
    '''
    ret = np.nancumsum(a, dtype=np.float32) # 누적합 배열을 만든다.        
    '''
    ret[:-n] 처음부터 시작, 끝에서 n개 뺀 것까지. (0 ~ len(arr)-n)
    
    ret[n:] = ret[n:] - ret[:-n]
    assume n = 5 :
      5 6 7 8 9 10 11 12 13 ...
    - 0 1 2 3 4  5  6  7  8 ...
    ------------------------
      5 5 5 5 5  5  5 ....

    여기서 ret가 누적합이므로, moving average값이 된다.
    '''
    ret[n:] = ret[n:] - ret[:-n] # window에 포함된 값들의 총합들을 구한다. n개의 합부터 시작
    return ret[n - 1:] / n

# As provided in the answer by Divakar
def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


# My modification to do a backward-fill
def bfill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


def fb_fill_1dim(arr):
    arr = arr.reshape((1, -1))
    arr = ffill(arr)
    arr = bfill(arr)
    arr = arr.reshape((-1))
    return arr

def min_to_sec(minval):
    return minval * 60

def to_sr(a):
    return sample_rate * a

in_wlen = to_sr(20)

def calc_hb_hr(segx):
    """extrace number of heartbeats in segment and heartbeat rate"""
    
    prominence_candidates = [20, 16, 12, 10]
    for prominence in prominence_candidates:
        peaks, _ = find_peaks(segx, distance=50, prominence=prominence)    
        heartbeats = len(peaks)
        diffs = np.diff(peaks)
        if len(diffs) == 0 or heartbeats == 0:
            continue            
        heartrates = round(in_wlen / np.diff(peaks).mean())
        return heartbeats, heartrates
    return 0, 0


def check_valid(segx):
    if np.mean(np.isnan(segx)) > 0.1 or \
        np.max(segx) > 200 or np.min(segx) < 20 or \
        np.max(segx) - np.min(segx) < 30 or \
        (np.abs(np.diff(segx[~np.isnan(segx)])) > 30).any():
        return False
    return True