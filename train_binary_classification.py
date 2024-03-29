import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
import datetime
from layers import GraphConvolution
import pickle
from scipy.sparse import csr_matrix
import torch.nn.init as init
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score


def kmax_pooling(x, dim, k):#kmax_pooling实现最大值池化操作,取出整个tensor中的前k个最大值。
    # x (torch.Tensor): 输入特征图
    #     dim (int): 池化操作在第dim维度进行
    #     k (int): 取出的最大值个数
    #     Returns:
    #     output (torch.Tensor): 最大值池化后的特征图
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    #     使用topk取出dim维度上的k个最大值的索引
    return x.gather(dim, index)
    # 根据索引进行gather操作,提取最大值特征图


def get_sentense_marix(x):   #get_sentense_marix生成一个140x140的句子成语矩阵,其entries表示句子中不同词汇的关系。
    # Args:
    # x(list): 句子中词汇的下标序列, 长度不超过140个词
    # Returns:
    # sentence_matrix(torch.FloatTensor): 140
    # x140的句子成语矩阵
    one_matrix = np.zeros((140, 140), dtype=np.float32) #生成140x140的零矩阵
    for index, item in enumerate(x):
        one_matrix[index][index] = 1    #主对角线置1,表示每个词与自身的关系
        if not item:                    #该词没有后继词
            one_matrix[index, item-1] = 2  #前一词与当前词关系为2
            one_matrix[item-1, index] = 3  #当前词与前一词关系为3
    return torch.FloatTensor(one_matrix)   #转为torch.FloatTensor tensor
#该函数实现了生成句子成语矩阵的功能。给定句子中每个词的下标,
#生成140x140的矩阵,表示句子中任意两个词的关系(前后、前后隔一、无关系)。
#使用numpy和torch进行高效 tensor 操作实现。




# h.p. define #Hyperparameters 超参数定义
torch.manual_seed(1) #控制随机数生成器的随机种子,确保结果复现性
EPOCH = 200          #训练epochs数
BATCH_SIZE = 32      #训练batch size
LR = 0.001           #学习率
HIDDEN_NUM = 64      #隐藏层神经元数
HIDDEN_LAYER = 2     #隐藏层数
# process data
print("Loading data...")
max_document_length = 140 #最大句子长度

fr = open('data_train_noRen_noW2v.txt', 'rb') #打开训练数据文件
#从文件中加载训练数据
x_train = pickle.load(fr)   #训练句子下标序列
y_train = pickle.load(fr)   #训练句子对应的标签
length_train = pickle.load(fr) #每个训练句子的长度

#pickle.load():从 opened 文件中加载数据,依次为句子下标序列、对应标签和句子长度。
fr = open('data_test.txt', 'rb')
x_dev = pickle.load(fr)
y_dev = pickle.load(fr)
length_dev = pickle.load(fr)
# Randomly shuffle data #随机打乱数据
np.random.seed(10)    #控制打乱操作的随机种子,确保结果的不确定性
shuffle_indices = np.random.permutation(np.arange(len(y_train)))  #得到长度为训练数据大小的随机打乱序列下标
print(shuffle_indices.shape) #打乱后下标序列的形状
print('x_train shape ', x_train.shape) #训练句子下标特征形状
#该部分实现了训练数据的随机打乱,旨在打乱原有的训练数据序而产生更加独立 reciprocally 的训练实例,
#进而提高模型对新数据的泛化能力与健壮性。

x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]

length_shuffled_train = length_train[shuffle_indices] #根据打乱序重新排列训练句子长度
#将 NumPy 训练特征和目标转为 PyTorch tensors
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).long()
#处理验证集
length_dev = [] #验证集句子长度
for item in x_dev: #逐句检查
    length_dev.append(list(item).index(0)) #记录0索引处,即句尾
print(len(length_dev)) #检查长度列表长度
x_dev = torch.from_numpy(x_dev) #将 NumPy 验证集特征转为 PyTorch tensors
y_dev = torch.max(torch.from_numpy(y_dev).long(), dim=1)[1]


train_x,train_y=[],[]  #训练集特征与对应目标
for x,y in zip(x_train, y_train):  #依次取出训练句子特征与对应的目标标签
    y = list(y).index(1)           #找到目标中 1 的下标,表示对应分类
    if y <=3:                      #如果属于第1、2、3个分类
        train_y.append(torch.unsqueeze(torch.FloatTensor([1.0,0.0]),dim=0))  #创建对应分类的one-hot vector,追加到目标
        train_x.append(torch.unsqueeze(x,dim=0))    #将句子句子特征扩充维度,追加到特征
    elif 6>y>3:
        train_y.append(torch.unsqueeze(torch.FloatTensor([0.0,1.0]),dim=0))
        train_x.append(torch.unsqueeze(x,dim=0))

#依次取出句子特征与原有目标
# 找到目标中 1 的下标,获取对应分类
# 根据分类范围创建对应one-hot vector
# 将句子特征扩充维度,与one-hot vector 追加到训练特征与目标


#   y = torch.LongTensor(y)
#print(train_y[0].shape)
train_x = torch.cat(train_x, dim=0)  #将追加的句子特征在第一维度上拼接
train_y = torch.cat(train_y, dim=0).long()    #将对应的one-hot 目标拼接,再转为LongTensor
#print(train_y)

#print(train_y.shape)


test_x,test_y=[],[]
for x,y in zip(x_dev, y_dev):
    y = y.data.item()
    if y <=3:
        test_y.append(0)
        test_x.append(torch.unsqueeze(x,dim=0))
    elif 6>y>3:
        test_y.append(1)
        test_x.append(torch.unsqueeze(x,dim=0))
#该部分实现了将原有验证集的多分类标记,转换为0、1 表示,
#以作为模型评估的输入。

#具体实现:

# 依次取出句子特征与原有标记
# 获取标记的Ordinal 表示
# 根据分类范围创建0或1目标值
# 将句子特征扩充维度,与目标值追加到验证特征与目标



#   y = torch.LongTensor(y)
#该部分实现了将预处理得到的验证特征与对应目标在第一维度上拼接,
#并将目标转换为合适类型,形成验证数据的最终形式,作为模型评估的输入。
test_x = torch.cat(test_x, dim=0)
test_y = torch.LongTensor(test_y)

#该部分实现了构建训练与验证TensorDataset,并创建训练dataloader,
#作为模型训练与评估的输入数据源。
torch_dataset = Data.TensorDataset(train_x, train_y) #训练数据的TensorDataset
torch_testset = Data.TensorDataset(test_x, test_y)   #验证/测试数据的TensorDataset
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
#DataLoader():基于提供的TensorDataset构建一个数据dataloader。 采用可选参数batch_size控制批量大小, shuffle进行打乱。

test_loader = Data.DataLoader(
    dataset=torch_testset,
    batch_size=128
)
print("data process finished")
#该部分实现了构建验证dataloader,
#作为模型评估输入数据源。



#print(x_train.shape)
#fear, anger, disgust,sadness, happy,like,surprise

class LSTM_GCN(nn.Module):
    def __init__(self):
        # 定义LSTM_GCN网络结构,包含视觉embedding层、双向LSTM网络、1D Batch Norm层、Graph Convolution层。
        super(LSTM_GCN, self).__init__()   #继承父类的构造方法
        self.embedding = nn.Embedding(76215, 300).cuda()   #视觉嵌入层,创建视觉词嵌入矩阵 size:76215 x 300
        self.lstm = nn.LSTM(  #LSTM网络,包含2层
            input_size=300,  #词嵌入大小为300
            hidden_size=180,  #输出隐单元大小为180
            num_layers=2,  #有2层LSTM单元
            batch_first=True,
            dropout=0.5,  #双向LSTM
            bidirectional=True
        ).cuda()
        self.batch1 = nn.BatchNorm1d(max_document_length).cuda() #添加1D Batch Normalization层
        self.gc = GraphConvolution(360, 2)  ##Graph Convolution层,输入维度360,输出维度2
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=1)
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=1)
        init.xavier_normal_(self.lstm.all_weights[1][0], gain=1)
        init.xavier_normal_(self.lstm.all_weights[1][1], gain=1)

    def forward(self, x_and_adj):
        x = x_and_adj[:, :max_document_length].cuda()   #取输入的第一个max_document_length个句子作为文本特征
        adj = x_and_adj[:, -max_document_length:]      #取输入后续的句子adjacency matrix作为关系特征
        x = self.embedding(x)  #将文本特征送入视觉嵌入层
        lstm_out, _ = self.lstm(x, None)   #送入LSTM网络,返回LSTM最后一个时间步的输出及隐单元状态
        out = self.batch1(lstm_out)     #将LSTM输出送入Batch Normalization层
        out = F.relu(out)  #ReLU激活function
        adj_Metrix = []  #创建用于存放各个句子adj matrix的列表
        for item in adj:
            adj_Metrix.append(torch.unsqueeze(get_sentense_marix(item), dim=0))
        adj_Metrix = torch.cat(adj_Metrix, dim=0)
        out_g1 = self.gc(out, adj_Metrix)
        out = torch.median(out_g1, 1)[0]
        return out
    # 定义整个网络的前向传播过程。 其中:
    # x_and_adj:包含输入的文本特征与关系特征的tensor
    # x:取输入的前max_document_length个句子作为文本特征
    # adj:取输入后续的句子adjacency matrix作为关系特征
    # embedding层将文本特征送入嵌入空间
    # LSTM层获取双向LSTM网络的输出
    # Batch Norm层减少内部covar变化,提高训练稳定性
    # 为每个句子的 adj matrix 添加第0维扩展,并在第0维拼接,构建关系特征
    # Graph Convolution层将文本特征与关系特征融合,获取输出
    # 取输出的中值作为最终结果


model = LSTM_GCN() # 创建LSTM_GCN模型实例
model.train()    # 将模型转为训练模式
# 创建Adam优化器,学习率为0.001,权重衰减为1e-8
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
loss_func = nn.BCEWithLogitsLoss()  #创建二分类损失,适用于两个类别
print(model)
best = 0


def get_test():
    global best  #引用best与更新
    model.eval()  #将模型转为评估模式,weights=false
    print('start dev test')
    record = []   #创建 predictions list,用于保存模型预测结果
    for index, (batch_x, batch_y) in enumerate(test_loader): #对每一批次输入进行测试
        test_output = model(batch_x)  #获取模型输出
        test_output = list(torch.max(test_output, dim=1)[1].cpu().numpy())  #取输出的最大值,将tensor变为list
        record.extend(test_output) #将这一批次的预测结果添加到record
    label = list(test_y.numpy()) #将label转为list
    y_true = label #将真实 label 值赋予y_true
    y_pred = record #将模型预测结果赋予y_pred
    print(len(y_true))
    print(len(y_pred))

    print("accuracy:", accuracy_score(y_true, y_pred))   #计算准确率,返回正确分类的样本数
    if accuracy_score(y_true, y_pred) > best: #如果准确率超过best,保存模型
        torch.save(model, "best_model.pth")
    print("macro_precision", precision_score(y_true, y_pred, average='macro'))  #计算宏平均精度
    print("micro_precision", precision_score(y_true, y_pred, average='micro'))

    # Calculate recall score
    print("macro_recall", recall_score(y_true, y_pred, average='macro'))
    print("micro_recall", recall_score(y_true, y_pred, average='micro'))

    # Calculate f1 score
    print("macro_f", f1_score(y_true, y_pred, average='macro'))
    print("micro_f", f1_score(y_true, y_pred, average='micro'))

    model.train()


f = open('accuracy_record.txt', 'w+')
f2 = open('loss_record.txt', 'w+')
loss_sum = 0
accuracy_sum = 0

for epoch in range(EPOCH):
    for index, (batch_x, batch_y) in enumerate(loader):
        right = 0
        if index == 0:
            get_test()
            loss_sum = 0
            accuracy_sum = 0
        #   one hot to scalar
        batch_y = batch_y.cuda()
        output = model(batch_x)
        optimizer.zero_grad()
        output = output.cuda()
       # print(output.shape)
        batch_y = batch_y.float()
        loss = loss_func(output, batch_y)
        #   gcnloss = ((torch.matmul(model.gc.weight.t(), model.gc.weight) - i)**2).sum().cuda()
        #   loss += gcnloss * 0.000005
        lstmloss = 0
        for item in model.lstm.parameters():
            if len(item.shape) == 2:
                I = torch.eye(item.shape[1]).cuda()
                lstmloss += ((torch.matmul(item.t(), item)-I)**2).sum().cuda()
        loss += lstmloss * 0.00000005
        loss.backward()
        predict = torch.argmax(output, dim=1).cpu().numpy().tolist()
        label = batch_y.cpu().numpy().tolist()

        for i in range(0, batch_y.size(0)):
            if predict[i] == label[i].index(1.0):
                right += 1
        optimizer.step()
        accuracy_sum += right/batch_y.size(0)
        loss_sum += float(loss)
        if index % 50 == 0:
            print("batch", index, "/ "+str(len(loader))+": ",  "\tloss: ", float(loss), "\taccuracy: ", right/batch_y.size(0))
    print('epoch: ', epoch, 'has been finish')
