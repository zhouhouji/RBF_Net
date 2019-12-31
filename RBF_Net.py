# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 16:04
# @Author  : Zhou
# @FileName: RBF_Net.py
# @Software: PyCharm
import numpy as np

class RBF_Net:
    def __init__(self,feature, label, n_hidden, maxCycle, n_output, learn_rate=0.01):
        self.feature = feature
        self.label = label
        self.n_hidden = n_hidden
        self.maxCycle = maxCycle
        self.n_output = n_output
        self.learn_rate =learn_rate

    def EuclideanDist(self,feature_i, center):
        m, n = center.shape
        EDist = np.zeros((m, 1))
        for i in range(m):
            EDist[i, 0] = (feature_i - center[i, :]) * (feature_i - center[i, :])
        return EDist

    def InitialCenter(self,k, dataset):
        '''
        :param row_num: the number of rows
        :param col_num: the number of columns
        :return: Conductance: the Conductance of array
        '''
        n = dataset.shape[1]  # columns
        clustercents = np.mat(np.zeros((k, n)))  # initialize centers
        for col in range(n):
            mincol = np.min(dataset[:, col])
            maxcol = np.max(dataset[:, col])
            # random.rand(k, 1):产生一个0~1之间的随机数向量（k,1表示产生k行1列的随机数）
            clustercents[:, col] = np.mat(mincol + float(maxcol - mincol) * np.random.rand(k, 1))  # 按列赋值
        return clustercents

    def hidden_out(self,x, sigma,func="Guess"):
        if func == "Guess":
            y = -1.0 * (np.multiply(x, x)) / (2 * np.multiply(sigma.T, sigma.T))
            hid_out = np.exp(y)
        if func == "Sigmoid":
            pass
        return hid_out

    def predict_in(self,hid_out, w):
        return hid_out * w

    def predict_out(self,x):
        '''输出层神经元激活函数
        线性激活
        '''
        return x

    def get_cost(self,cost):
        '''计算当前损失函数的值
        input:  cost(mat):预测值与标签之间的差
        output: cost_sum / m (double):损失函数的值
        '''
        m, n = cost.shape
        cost_sum = 0.0
        for i in range(m):
            for j in range(n):
                cost_sum += cost[i, j] * cost[i, j]
        return cost_sum / 2

    def get_predict(self,feature, center, sigma, w):
        '''计算最终的预测
        input:  feature(mat):特征
                w0(mat):输入层到隐含层之间的权重
                b0(mat):输入层到隐含层之间的偏置
                w1(mat):隐含层到输出层之间的权重
                b1(mat):隐含层到输出层之间的偏置
        output: 预测值
        '''
        row_num = feature.shape[0]
        output_out = np.zeros((row_num, 1))
        for j in range(row_num):
            # 2.1、信号正向传播
            # 2.1.1、计算隐含层的输出
            EDist = self.EuclideanDist(feature[j, :], center)
            hidden_output = self.hidden_out(EDist, sigma)
            # 2.1.3、计算输出层的输入
            output_in = self.predict_in(hidden_output, w)
            # 2.1.4、计算输出层的输出
            output_out[j, 0] = self.predict_out(output_in)
        return output_out

    def save_model_result(self,center, delta, w, result):
        '''保存最终的模型
        input:  w0(mat):输入层到隐含层之间的权重
                b0(mat):输入层到隐含层之间的偏置
                w1(mat):隐含层到输出层之间的权重
                b1(mat):隐含层到输出层之间的偏置
        output:
        '''

        def write_file(file_name, source):
            f = open(file_name, "w")
            m, n = source.shape
            for i in range(m):
                tmp = []
                for j in range(n):
                    tmp.append(str(source[i, j]))
                f.write("\t".join(tmp) + "\n")
            f.close()

        write_file("center.txt", center)
        write_file("delta.txt", delta)
        write_file("weight.txt", w)
        write_file('train_result.txt', result)

    def err_rate(self,label, pre):
        '''计算训练样本上的错误率
        input:  label(mat):训练样本的标签
                pre(mat):训练样本的预测值
        output: rate[0,0](float):错误率
        '''
        m = label.shape[0]
        for j in range(m):
            if pre[j, 0] > 0.5:
                pre[j, 0] = 1.0
            else:
                pre[j, 0] = 0.0

        err = 0.0
        for i in range(m):
            if float(label[i, 0]) != float(pre[i, 0]):
                err += 1
        rate = err / m
        return rate

    def Train(self,feature, label, n_hidden, maxCycle, alpha, n_output):
        '''计算隐含层的输入
        input:  feature(array):特征
                label(array):标签
                n_hidden(int):隐含层的节点个数
                maxCycle(int):最大的迭代次数
                alpha(float):学习率
                n_output(int):输出层的节点个数
        output: center(array):rbf函数中心
                delta(array):rbf函数扩展常数
                w(array):隐含层到输出层之间的权重
        '''
        # 数据初始化
        sigma = np.mat(np.random.rand(n_hidden, 1))
        center = self.InitialCenter(n_hidden, feature)
        w = np.mat(np.random.rand(n_hidden, n_output))

        row_num = feature.shape[0]
        output_out = np.zeros((row_num, 1))
        EDist = np.mat(np.zeros((row_num, n_hidden)))
        hidden_output = np.mat(np.zeros((row_num, n_hidden)))
        cost = np.inf

        # 训练过程
        item = 0
        while item < maxCycle:
            for j in range(row_num):
                # 2.1、信号正向传播
                # 2.1.1、计算隐含层的输出
                # 计算第j个样本与center之间的欧式距离
                EDist[j, :] = self.EuclideanDist(feature[j, :], center)
                # 用高斯函数激活
                hidden_output[j, :] = self.hidden_out(EDist[j, :], sigma)
                # 2.1.3、计算输出层的输入
                output_in = self.predict_in(hidden_output[j, :], w)
                # 2.1.4、计算输出层的输出，线性
                output_out[j, 0] = self.predict_out(output_in)

            # 2.2、误差的反向传播
            error = label - output_out

            for j in range(n_hidden):
                # sum1 = 0.0
                sum2 = 0.0
                sum3 = 0.0

                for i in range(row_num):
                    sum2 += error[i, :] * hidden_output[i, j] * EDist[i, j]
                    sum3 += error[i, :] * hidden_output[i, j]

                delta_sigma = (w[j, :] / (sigma[j, 0] * sigma[j, 0] * sigma[j, 0])) * sum2
                delta_w = sum3
                # 2.3、 修正权重和rbf函数中心和扩展常数

                sigma[j, 0] = sigma[j, 0] + alpha * delta_sigma
                w[j, :] = w[j, :] + alpha * delta_w

            item += 1
            if item % 10 == 0:
                cost = (1.0 / 2) * self.get_cost(self.get_predict(feature, center, sigma, w) - label)
                print("\t-------- item: ", item, " ,cost: ", cost)
            if cost < 5:  ###如果损失函数值小于5则停止迭
                break
        return center, sigma, w


    def Start_train(self):
        center, delta, w = self.Train(
            self.feature,self.label,self.n_hidden,self.maxCycle,self.n_output,self.learn_rate
        )
        result = self.get_predict(self.feature, center, delta, w)
        print("训练准确性为：", (1 - self.err_rate(self.label, result)))
        self.save_model_result(center, delta, w, result)
        print("train has been saved!\n")
        print("model train has completed")
