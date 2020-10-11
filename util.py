import numpy as np

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
    

