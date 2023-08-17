import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ANN():
    def __init__(self,layer_number,layer_neuron,learning_rate,batch_size=20,epochs=100):
        self.layer_number = layer_number
        self.layer_neuron = layer_neuron
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-1 * x))

    def mean_square_loss(self,s, y):
        return np.sum(np.square(s - y) / 2)

    def miniBatch(self,X, Y):
        sample_num = X.shape[0] # 输入训练的样本数量
        idx = list(range(sample_num))
        np.random.shuffle(idx)
        Xbatchs = list()
        Ybatchs = list()
        batch_num = int(sample_num / self.batch_size) # 一个epoch中的batch数量
        if(batch_num == 0):
            raise Exception("sample_num < batch_size")
        for i in range(batch_num):
            I = idx[i * self.batch_size:(i + 1) * self.batch_size]
            Xbatchs.append(X[I, :])
            Ybatchs.append(Y[I, :])
        return Xbatchs, Ybatchs, batch_num


    def randomInitWB(self):
        # input_neuron = self.layer_neuron[0]
        Wlist = list()
        Blist = list()
        # Wlist.append(np.random.rand(input_neuron, self.layer_neuron[1]))
        # Blist.append(np.random.rand(1, self.layer_neuron[1]))
        for i in range(self.layer_number - 1):
            Wlist.append(np.random.normal(0.0,pow(self.layer_neuron[i+1],-0.5),(self.layer_neuron[i], self.layer_neuron[i + 1])))
            Blist.append((np.random.normal(0.0,pow(self.layer_neuron[i+1],-0.5), (1,self.layer_neuron[i + 1]))))
        return Wlist, Blist


    def forward(self,X, Y, Wlist, Blist):
        alist = list() # alist同来存储当前隐层的输出，下一个隐层的输入
        zlist = list() # zlist同来存储当前隐层激活前的数据，即权重偏置作用的结果
        # 输入层的处理有些许不同
        z_temp = np.matmul(X, Wlist[0]) - Blist[0] # 权重偏置结果
        zlist.append(z_temp)
        alist.append(self.sigmoid(z_temp)) # 激活
        for i in range(1,self.layer_number - 1):
            z_temp = np.matmul(alist[i-1], Wlist[i]) - Blist[i] # 权重偏置结果
            zlist.append(z_temp)
            alist.append(self.sigmoid(z_temp)) # 激活
        loss = self.mean_square_loss(alist[self.layer_number - 2], Y)# 误差值
        return zlist, alist, loss

    def transform(self,X,Wlist,Blist):
        alist = list()  # alist同来存储当前隐层的输出，下一个隐层的输入
        zlist = list()  # zlist同来存储当前隐层激活前的数据，即权重偏置作用的结果
        # 输入层的处理有些许不同
        z_temp = np.matmul(X, Wlist[0]) - Blist[0]  # 权重偏置结果
        zlist.append(z_temp)
        alist.append(self.sigmoid(z_temp))  # 激活
        for i in range(1, self.layer_number - 1):
            z_temp = np.matmul(alist[i - 1], Wlist[i]) - Blist[i]  # 权重偏置结果
            zlist.append(z_temp)
            alist.append(self.sigmoid(z_temp))  # 激活
        return alist[-1]

    def transform(self,X):
        alist = list()  # alist同来存储当前隐层的输出，下一个隐层的输入
        zlist = list()  # zlist同来存储当前隐层激活前的数据，即权重偏置作用的结果
        # 输入层的处理有些许不同
        z_temp = np.matmul(X, self.Wlist[0]) - self.Blist[0]  # 权重偏置结果
        zlist.append(z_temp)
        alist.append(self.sigmoid(z_temp))  # 激活
        for i in range(1, self.layer_number - 1):
            z_temp = np.matmul(alist[i - 1], self.Wlist[i]) - self.Blist[i]  # 权重偏置结果
            zlist.append(z_temp)
            alist.append(self.sigmoid(z_temp))  # 激活
        return alist[-1]

    def backward(self,X, Y, Wlist, Blist):
        zlist, alist, loss = self.forward(X, Y, Wlist, Blist)
        dWlist = list() # 各层间W的梯度的反方向
        dBlist = list() # 各层间b的梯度的反方向
        # 最后层与倒数第二层间的权重与偏置的梯度
        a1 = alist[self.layer_number - 2] # 输出层的输出（y_hat)
        a2 = alist[self.layer_number - 3] # 输出层的输入/上一层的输出
        g = a1 * (1 - a1) * (a1 - Y)
        dW = - np.matmul(a2.T, g)
        db = np.sum(g, axis=0)
        dWlist.append(dW)
        dBlist.append(db)

        # 除第一层和最后一层外，中间的权重与偏置的更新
        for i in range(1, self.layer_number - 2).__reversed__():
            a1 = alist[i]
            a2 = alist[i - 1]
            w = Wlist[i + 1]
            g = a1 * (1 - a1) * (np.matmul(g, w.T))
            dW = - np.matmul(a2.T, g)
            db = np.sum(g, axis=0)
            dWlist.append(dW)
            dBlist.append(db)
        # 第一层与第二层间的权重与偏置的梯度
        a1 = alist[0]
        a2 = X
        w = Wlist[1]
        g = a1 * (1 - a1) * (np.matmul(g, w.T))
        dW = -np.matmul(a2.T, g)
        db = np.sum(g, axis=0)
        dWlist.append(dW)
        dBlist.append(db)

        return dWlist, dBlist, loss


    # 多层神经网络
    def fit(self,X, Y):
        # 随便创的数据和权重，偏置值，小伙伴们也可以使用numpy的ranodm()进行随机初始化
        iternum = 0

        input_neuron = self.layer_neuron[0]
        # 随机初始化每层的权重与偏置
        Wlist, Blist = self.randomInitWB()

        for e in range(self.epochs):
            # 将原数据集拆分mini Batch
            Xbatchs, Ybatchs, batch_num = self.miniBatch(X, Y)
            for i in range(batch_num):
                dWlist, dBlist, loss = self.backward(Xbatchs[i], Ybatchs[i], Wlist, Blist)
                iternum = iternum + 1
                if (iternum % 5000 == 0):
                    print(f'第{iternum}次迭代 loss={loss}')
                    # print(dWlist)

                for j in range(self.layer_number-1):
                    Wlist[j] = Wlist[j] + self.learning_rate * dWlist[self.layer_number - j - 2]
                    Blist[j] = Blist[j] + self.learning_rate * dBlist[self.layer_number - j - 2]
            self.Wlist = Wlist
            self.Blist = Blist
            zlist, alist, loss = self.forward(X, Y, Wlist, Blist)
        return Wlist, Blist, loss


# file_name = "E:/pycharm files\机器学习/neural-network-master/neural-network-master/mnist_train/mnist_train.csv"
# df = pd.read_csv(file_name,header=None)
# df_image = df.iloc[:,1:]
# df_result = df.iloc[:,[0]]
#
#
# df_onehot = pd.get_dummies(df_result,prefix_sep='_',columns=df_result.columns)
#
#
# X = df_image.to_numpy()
# y = df_onehot.to_numpy()
#
# X = X/255.0 *0.99 + 0.01
# y = y *0.99
#
#
#
#
# ann = ANN(layer_number=3,
#           layer_neuron=[784,200,10],
#           learning_rate=0.05,
#           batch_size=15,
#           epochs=5)
#
#
# a = ann.train(X,y)
# print(f"loss={a[4]}")
