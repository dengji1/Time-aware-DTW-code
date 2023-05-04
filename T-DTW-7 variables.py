import os
import pandas as pd
import numpy as np
import math
import ctypes
from sklearn.cluster import SpectralClustering
from multiprocessing import Process, Array
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from scipy import stats

#this code calculate distance matrices using T-DTW and DTW, and performs clustering
#on the first 5000 time series of seven selected variables.

#max value and min value of variables in the first 5000 files
max_dict = {'HR':223,'SBP':274,'DBP':287,'O2Sat':100,'Resp':69,'MAP':298,'Temp':41.22}
min_dict = {'HR':20 ,'SBP':24,'DBP':20,'O2Sat':22,'Resp':1,'MAP':20,'Temp':20.9}

#get data files
def get_files(file_path):
    files = []

    for base_path in file_path:
        for f in os.listdir(base_path):
            full_path = os.path.join(base_path, f)
            files.append(full_path)

    files = sorted(files)[0:5000]
    print('number of files: {}'.format(len(files)))
    return files

#extract time series and labels
def get_ts_label(files, variable):
    ts=[]
    label=[]
    for f in files:
        df = pd.read_table(f,sep='|',header=0,index_col=False)
        s = df.loc[:,variable]
        lb = df.loc[:,'SepsisLabel']
        if not s.isnull().all(): #exclude time series with only missing values
            ts.append(s)
            label.append(1 if lb.sum() else 0)

    print('number of time series: {}'.format(len(ts)))
    return ts, label

#present time series as [[value1, interval1],[value2,interval2]...]
def process_ts(ts, variable):
    processed_ts = []
    for s in ts:
        s1 = s[s.notnull()] #throw out missing values
        values = np.array(s1).reshape(-1,1) 

        t = list(s1.index)
        t_cur = np.array(t).reshape(-1,1)
        t.insert(0,0)
        t_pre = np.array(t[:-1]).reshape(-1,1)
        interval = t_cur - t_pre #sampling interval

        values_scaled = (values - min_dict[variable]) / (max_dict[variable] - min_dict[variable])#normalization

        p_s = np.concatenate([values_scaled,interval],axis=1)
        processed_ts.append(p_s)
    return processed_ts

#calculate distance between points
def dist(p1, p2, a):
    d = abs(p1[0] - p2[0]) + a * abs(p1[1] - p2[1])
    return d

#DTW
def DTW(s1, s2, a):
    len1 = len(s1)
    len2 = len(s2)
    D = np.full((len1+1,len2+1),math.inf)
    D[0][0] = 0

    for i in range(len1):
        for j in range(len2):
            d = dist(s1[i],s2[j],a)
            D[i+1][j+1] = d + min([D[i,j+1],D[i,j],D[i+1,j]])
    return D[len1][len2]

#function run by each process
def worker(base_array, rows, processed_ts, a):
    num = len(processed_ts)
    dist_matrix = np.frombuffer(base_array, dtype=ctypes.c_double).reshape((num,num))
    for i in rows:
        for j in range(i):
            dist_matrix[i][j] = DTW(processed_ts[i],processed_ts[j],a)


#calculate distance matrix
def get_dist_matrix(processed_ts, a, process_num):
    num = len(processed_ts)
    base_array = Array(ctypes.c_double, np.zeros((num*num,)), lock=False) #the distance matrix lies in shared memory

    #assign matrix rows to calculate for each process
    p_rows = {}
    for i in range(process_num):
        p_rows[i] = []
    r = 0
    while r < (num + 1) // 2:
        for i in range(process_num):
            if r >= (num + 1) // 2:
                break
            p_rows[i].append(r)
            p_rows[i].append(num-1-r)
            r += 1

    processes = []
    for i in range(process_num):
        p = Process(target=worker, args=(base_array, p_rows[i], processed_ts, a))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print('all {} processes finished'.format(process_num))

    dist_matrix = np.frombuffer(base_array, dtype=ctypes.c_double).reshape((num,num))
    dist_matrix = dist_matrix + dist_matrix.T

    return dist_matrix

#print clustering result and chi-square value
def get_result(pred, label):
    df = pd.DataFrame({'pred':pred,'label':label})
    g = df.groupby('pred')
    tot = list(g.count().values.reshape(-1))
    sepsis = list(g.sum().values.reshape(-1))
    non_sepsis = list((g.count()-g.sum()).values.reshape(-1))
    result = pd.DataFrame({'sepsis':sepsis,'non_sepsis':non_sepsis,'all':tot})
    sep = np.array(sepsis).reshape(-1,1)
    non_sep = np.array(non_sepsis).reshape(-1,1)
    bg = np.concatenate([sep, non_sep],axis=1)
    
    chi2, p, dof, ex = stats.chi2_contingency(bg,correction=False)
    print('chi2={},p={}'.format(chi2, p))
    print(result)
    


if __name__ == '__main__':

    #paths of data
    file_path = ['/home/dengji/Data/training_setA','/home/dengji/Data/training_setB']
    #variables
    variable_list = ['HR','MAP','SBP','DBP','Resp','O2Sat','Temp']
    #hyperparameter a
    a_list = [0, 0.01,0.02,0.03,0.04,0.05]
    #number of processes
    process_num = 80

    for variable in variable_list:
        for a in a_list:
            files = get_files(file_path)
            ts,label = get_ts_label(files, variable)
            processed_ts = process_ts(ts, variable)
            dist_matrix = get_dist_matrix(processed_ts, a, process_num)

            #transform distance matrix into similarity matrix by Gaussian kernel
            delta = 20
            sim_matrix = np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

            #perform spectral clustering
            sc = SpectralClustering(4, affinity='precomputed', assign_labels='cluster_qr')
            pred = sc.fit_predict(sim_matrix)

            #print result and metrics
            print(variable,'a={}'.format(a))
            get_result(pred, label)
            print('adjusted_rand_score={}'.format(adjusted_rand_score(pred, label)))
            print('mutual_info_score={}'.format(mutual_info_score(pred, label)))
            print('adjusted_mutual_info_score={}'.format(adjusted_mutual_info_score(pred, label)))
            print('normalized_mutual_info_score={}'.format(normalized_mutual_info_score(pred, label)))
            print('homogeneity_score={}'.format(homogeneity_score(pred, label)))
            print('completeness_score={}'.format(completeness_score(pred, label)))