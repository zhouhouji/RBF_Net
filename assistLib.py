import pandas as pd
import numpy as np

#read csv file to array
def readFiles_csv(filename):
    rawData = pd.read_csv(filename)
    rawData = rawData.values
    return rawData

# read txt files to array
def readFiles_txt(filename):
    '''导入数据
        input:  file_name(string):文件的存储位置
        output: feature_data(mat):特征
                label_data(mat):标签
                n_class(int):类别的个数
        '''
    # 1、获取特征
    f = open(filename)  # 打开文件
    feature_data = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    n_output = 1

    return np.mat(feature_data), np.mat(label).transpose(), n_output