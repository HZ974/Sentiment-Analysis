import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.init as init

class SparseMM(torch.autograd.Function):  #SparseMM算子功能
    def init(self, sparse):   #构造方法,负责将稀疏特征值特征名称参数化
        super(SparseMM, self).init()
        self.sparse = sparse.cuda() #将它上传到gpu上面

    @staticmethod  #前向传播方法
    def forward(cls, dense):
        return torch.bmm(cls.sparse, dense) #两次次稠密矩阵相乘

    @staticmethod   #后向传播方法,用于根据误差计算梯度
    def backward(cls, grad_output):
        grad_input = None  #用于存储输入梯度
        sparse_t = []     #用于存储稀疏特征值的转置
        for item in cls.sparse:  #将所有稀疏特征值转置
            sparse_t.append(torch.unsqueeze(item.t(), dim=0))
        sparse_t = torch.cat(sparse_t, dim=0)  #将所有稀疏特征值转置结果拼接
        if cls.needs_input_grad[0]:    #如果输入需要梯度
            grad_input = torch.bmm(sparse_t, grad_output)  #通过稀疏特征值转置和误差相乘来获取输入梯度
        return grad_input


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  #输入特征数
        self.out_features = out_features #输出特征数
        self.weight = Parameter(torch.Tensor(in_features, out_features)).cuda() #权重参数,划分到GPU
        if bias: #是否使用偏移
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):   #权值初始化
        stdv = 1. / math.sqrt(self.weight.size(1))  #标准差
        init.xavier_uniform(self.weight.data, gain=1) #使用Xavier方式初始化权重
        self.weight.data.uniform_(-stdv, stdv)#随机
        if self.bias is not None:
            init.xavier_uniform(self.bias.data, gain=1)
            self.bias.data.uniform_(-stdv, stdv)#随机

    def forward(self, input, adj):
        #   []
        weight_matrix = self.weight.repeat(input.shape[0], 1, 1) #将权重展开
        support = torch.bmm(input, weight_matrix)  #输入与权重相乘
        #print(adj.shape)
        #print(type(adj))
        output = SparseMM(adj)(support) #使用稀疏矩阵相乘
        if self.bias is not None: #是否使用偏移
            return output + self.bias.repeat(output.size(0))
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
