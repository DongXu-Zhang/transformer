# transformer
It's a brief introduction of transformer, which includes several basic concepts and operations.
## 1.1 Architecture of transformer
Transformer 架构是一种基于注意力机制的深度学习模型，由 Vaswani 等人在 2017 年提出。它颠覆了传统的 RNN 和 CNN，通过全局自注意力机制来捕捉序列中各元素间的依赖关系，从而实现高效并行计算和长程依赖建模。
这里给出了经典的transformer的架构图：
![architecture](https://github.com/user-attachments/assets/47e27f51-61ae-4669-ab67-d6d4ac75006c)
Transformer 模型主要由四部分组成：
- **编码器（Encoder）**：由N个编码器层堆叠而成，每个编码器层由两个子层连接组成。第一个子层连接结构包括一个多头自注意力子层（Multi Head Attention）和规范化层（Layer Normalization）以及一个残差连接（Residual Model）；第二个子层连接结构包含一个前馈全连接子层（Feed Forward）和规范化层以及一个残差连接。将输入序列（例如一段文本）转换为连续的高维表示。
- **解码器（Decoder）**：同样由N个解码器堆叠而成，每个解码器层由三个子层连接构成，第一个子层连接结构包括一个多头自注意力子层（Mutil Head Self-attention）和规范化层以及残差连接，后两个子层结构与编码器相同。基于编码器输出以及之前生成的目标序列，逐步生成最终的输出（如翻译文本）。
- **输入部分（Input）**：包含Embedding层以及positional encoding。
- **输出部分（Output）**：包含线性层和softmax处理。
基于seq2seq架构的transformer模型可以完成NLP领域研究的典型任务，例如机器翻译、文本生成等。同时又可以构建预训练语言模型，用于不同任务的迁移学习。
接下来会对编码器、解码器以及输入和输出部分进行一个详细的介绍。
## 2.1 Input part
输入部分包括源文本的嵌入层和源文本的位置编码器以及目标文本的嵌入层和目标文本的位置编码器，一共四个小部分。
### 文本嵌入层的作用
文本嵌入层是将离散的文本符号转换为连续向量表示的关键组件，通过查找表或神经网络参数将每个词或子词单元映射到一个高维向量空间，使得相似语义的词在向量空间中相互接近，从而为后续的模型层提供可微、可优化的输入。在现代 NLP 模型中，嵌入层不仅捕捉词语的静态语义关系，还会在训练过程中动态更新向量表示，使模型能够针对具体任务不断调整词向量以获得更优性能。
在 Transformer 架构中，文本嵌入层通常与位置编码相结合，以弥补模型本身缺乏顺序信息的不足。嵌入层首先将输入的离散 token 转换为向量表示，然后与相应的位置编码相加，从而在保持词语语义信息的同时引入位置信息。这种设计既确保了模型能够感知序列中词语的顺序，又维持了嵌入向量在语义空间中的连续性和可训练性。
```bash
# This is a detailed example of input embedding.
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
#定义Embeddings类来实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        '''d_model:指词嵌入的维度 vocab:指词表的大小'''
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        '''该层的前向传播逻辑 所有层都会有这个函数
        参数x:Embedding层是首层 所以代表输入给模型的文本通过词汇映射以后的张量'''
        return self.lut(x) * math.sqrt(self.d_model)
d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[1,2,3,4],[5,6,7,8]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print("embr",embr)
print(embr.shape)

# The results are as follows:
embr tensor([[[  8.1698, -17.9067, -18.2930,  ...,  20.5587, -34.3979,   2.2324],
         [ 14.6471,  11.7492,  -1.3063,  ...,  16.8272,  17.8224, -27.0285],
         [-27.9950,  -9.8002,   6.7823,  ...,  12.2133,  13.5687,  11.5653],
         [ 15.9306,  16.4315, -20.9837,  ...,  13.7980, -51.4857,   4.0905]],

        [[ -9.8196, -19.7474,  15.3972,  ...,  -7.8003, -25.9976,  28.1036],
         [ 10.8901,  -8.4099,  11.9932,  ..., -28.9421,  25.8586,  -5.6181],
         [ -8.1165,  -4.8526, -32.4488,  ..., -13.8231, -38.0867, -11.5192],
         [ -5.7509,  14.8611, -21.8043,  ..., -22.2486, -16.3707,  48.7701]]],
       grad_fn=<MulBackward0>)
torch.Size([2, 4, 512])
```
### 位置编码器的作用
传统的序列模型，如RNN或LSTM，通过其递归结构天然地捕捉了数据中的时间或序列顺序信息。而Transformer模型则完全依赖自注意力机制，这使得它无法通过网络结构本身区分输入序列中各个元素的顺序。位置编码器通过将额外的位置信息注入到输入嵌入（embedding）中，使得模型能够了解每个元素在序列中的具体位置。这种位置信息可以采用固定的正弦和余弦函数形式，也可以使用可学习的参数进行训练。固定编码的优势在于其平滑的周期性特征和较好的推广能力，而可学习编码则能够更好地适应特定任务的需求。
在实际应用中，位置编码器不仅为Transformer提供了绝对位置信息，还在一定程度上帮助模型捕捉了相对位置关系，这对于诸如机器翻译、文本生成以及语义理解等任务来说至关重要。通过有效地结合输入特征与位置信息，模型能够更准确地识别词汇之间的关系和上下文，从而提升整体性能和泛化能力。
```bash
# This is a detailed example of positional encoding.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        #分别表示词嵌入的维度，dropout层的置零比率 句子的最大长度
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout) #实例化dropout层
        pe = torch.zeros(max_len, d_model) #初始化一个位置编码的矩阵
        position = torch.arange(0, max_len).unsqueeze(1) #初始化一个绝对位置矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2)* -(math.log(10000.0)/d_model)) #定义一个变换矩阵div_term 进行跳跃式初始化
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #将变化的矩阵分别进行奇偶的赋值
        pe = pe.unsqueeze(0) #将二维张量扩展到三维
        #将位置编码矩阵注册成模型的buffer 该buffer不是模型中的参数 不会随着优化器进行改变
        #注册成buffer以后 就可以在模型保存后重新加载的时候就可以把位置编码器和模型参数加载进来
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #x代表文本序列的词嵌入表示
        #pe编码过长 所以把第二个维度 也就是maxlen对应的维度缩小成x的句子对应的维度
        x= x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
d_model = 512
dropout = 0.1
max_len = 60

x =  embr
pe =  PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print(pe_result)
print(pe_result.shape)

# The results are as follows:
tensor([[[-31.9787, -22.4201,  -2.7243,  ...,  22.8682,  42.1275,  18.0581],
         [ 70.4338,   0.5847,  -2.7951,  ..., -18.9241, -28.7554,  13.3666],
         [  0.5606,   9.6777, -40.3975,  ...,   0.1787,   0.0000, -18.7898],
         [ -0.0000,   9.8950, -19.4691,  ...,  10.6174,   7.8436,  19.6449]],

        [[-17.8046, -15.8142,  -0.0000,  ..., -16.2114,  -2.5738,  -0.0000],
         [  5.2375,  -8.1234,  -9.3114,  ...,   7.9752,  -7.7164,  10.9748],
         [ 16.3702, -14.5719,   3.8670,  ..., -29.5858,  -9.5673,  -0.0000],
         [ 12.2456,  16.9847,  37.8606,  ..., -50.1137, -45.1080, -24.3606]]],
       grad_fn=<MulBackward0>)
torch.Size([2, 4, 512])
```
## 3.1 Encoder part
### 3.1.1 掩码张量
- - **什么是掩码张量**
在Transformer中，掩码张量是一种用于在计算注意力权重时屏蔽某些位置的张量。它通常以二值矩阵或张量的形式出现，其中的“1”或“True”表示该位置的信息是有效的，而“0”或“False”则表示该位置的信息应当被忽略。例如，在处理不同长度的句子时，为了避免模型在计算自注意力时考虑填充（padding）部分，我们会构造一个填充掩码，将填充位置屏蔽掉，从而确保模型只关注真实的输入信息。
- - **掩码张量的作用**
此外，掩码张量在自回归模型（如Transformer解码器）中也扮演着至关重要的角色。在这种场景下，通常需要使用因果掩码（causal mask），它可以阻止当前位置访问未来的信息，确保模型在生成下一个单词时只依赖于当前及之前的上下文。这不仅防止了信息泄露，也保持了生成过程的合理性和一致性。总之，掩码张量通过屏蔽不相关或不允许访问的信息，为Transformer模型提供了更高效、更准确的注意力机制。
```bash
# This is a detailed example of mask tensor.
import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

# 构建掩码张量的函数
def subsequent_mask(size):
    '''生成向后遮掩的掩码张量 参数size是掩码张量最后两个维度的大小 最后两维形成了一个方阵'''
    # 首先定义掩码张量的形状
    attn_shape = (1, size, size)
    # 使用np.ones方法向这个形状中添加元素1 形成上三角阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 最后将numpy类型转换为torch中的tensor 然后做一个三角阵的反转
    return torch.from_numpy(1 - subsequent_mask)

size = 5
sm = subsequent_mask(size)
print('sm=',sm)
# The results are as follows:
sm= tensor([[[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1]]], dtype=torch.uint8)
```
### 3.1.2 注意力机制
- - **什么是注意力机制**
核心思想是：在处理序列或集合中每个元素时，根据该元素与其它元素的相关程度动态分配“注意力”权重，从而让模型能够“聚焦”于最重要的信息。在自然语言处理的Transformer中，注意力机制不仅能够捕捉长距离依赖，还能并行计算，大大提升了序列建模的效率和效果。
```bash
# This is a detailed example of attention mechanism.
def attention(query, key, value, mask=None, dropout=None):
    # QKV表示注意力的三个输入向量 mask表示掩码张量
    # 首先把query的最后一个维度提取出来 代表的是词嵌入维度
    d_k = query.size(-1)
    # 按照注意力的计算公式 把QK进行矩阵乘法 再进行缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 判断是否使用掩码张量
    if mask is not None:
        # 使用masked_fill 方法
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对scores的最后一个维度进行softmax操作
    p_attn = F.softmax(scores, dim=-1)
    #判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    #最后完成和V的乘法
    return torch.matmul(p_attn, value), p_attn

query = key = value = pe_result
mask = Variable(torch.zeros(2, 4, 4))
attn, p_attn = attention(query, key, value, mask=mask)
print('attn', attn)
print(attn.shape)
print('p_attn', p_attn)
print(p_attn.shape)
# The results are as follows:
attn tensor([[[ 11.9263, -13.5503, -10.0237,  ...,  18.0915,  -3.3088,  15.5298],
         [ 11.9263, -13.5503, -10.0237,  ...,  18.0915,  -3.3088,  15.5298],
         [ 11.9263, -13.5503, -10.0237,  ...,  18.0915,  -3.3088,  15.5298],
         [ 11.9263, -13.5503, -10.0237,  ...,  18.0915,  -3.3088,  15.5298]],

        [[-20.2358,   7.6624,  22.7583,  ...,   7.7242,  -0.7902, -14.7818],
         [-20.2358,   7.6624,  22.7583,  ...,   7.7242,  -0.7902, -14.7818],
         [-20.2358,   7.6624,  22.7583,  ...,   7.7242,  -0.7902, -14.7818],
         [-20.2358,   7.6624,  22.7583,  ...,   7.7242,  -0.7902, -14.7818]]],
       grad_fn=<UnsafeViewBackward>)
torch.Size([2, 4, 512])
p_attn tensor([[[0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500]],

        [[0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500]]], grad_fn=<SoftmaxBackward>)
torch.Size([2, 4, 4])
```
### 3.1.3 多头注意力机制
- - **什么是多头注意力机制**
对于多头注意力机制的每个头，都是从词义层面分割输出的向量，也就是每一组都会获得一组QKV去进行注意力机制下计算，但是句子中的每个词的表示只会获得一部分，也就是只分割了最后一维的词嵌入向量。这就是所谓的多头，然后将每个头的获得的输入送到注意力机制当中，就形成了多头注意力机制。
- - **多头注意力的作用**
多头注意力机制的设计能够让每个注意力机制去优化每个词汇的不同特征部分，从而去均衡同一种注意力机制可能产生的偏差，让词义拥有更加多元的表达，从而提升模型的效果。
## 4.1 Decoder part
## 5.1 Output part
