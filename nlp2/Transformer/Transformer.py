import torch
from torch import nn
import numpy as np

d_k = 64  # K(=Q)维度
d_v = 64  # V维度
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义缩放点积注意力类
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 维度信息
        # Q K V [batch_size, n_heads, len_q/k/v, dim_q=k/v] (dim_q=dim_k)
        # attn_mask [batch_size, n_heads, len_q, len_k]

        # 计算注意力分数（原始权重） [batch_size, n_heads, len_q, len_k]
        # scores [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2) / np.sqrt(d_k))

        # 使用注意力掩码，将attn_mask中值为1的位置的权重替换为极小值
        scores.masked_fill_(attn_mask, -1e9)
        # 对注意力分数进行softmax
        # weights [batch_size, n_heads, len_q, len_k]
        weights = nn.Softmax(dim=-1)(scores)

        # 计算上下文向量（也就是注意力输出），是上下文信息的紧凑表示
        # context [batch_size, n_heads, len_q, dim_v]
        context = torch.matmul(weights, V)

        return context, weights  # 返回上下文向量和注意力分数


# 定义多头注意力类
d_embedding = 512  # Embedding的维度
n_heads = 8  # Multi-Head Attention中头的个数
batch_size = 3  # 每一批的数据大小


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)  # Q的线性变换层
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)  # K的线性变换层
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)  # V的线性变换层
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, Q, K, V, attn_mask):
        # Q K V [batch_size, len_q/k/v, embedding_dim]
        residual, batch_size = Q, Q.size(0)  # 保留残差连接
        # 将输入进行线性变换和重塑，以便后续处理
        # q_s k_s v_s [batch_size, n_heads, len_q/k/v, d_q=k/v]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 将注意力掩码复制到多头
        # attn_mask [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 使用缩放点积注意力计算上下文和注意力权重
        # context [batch_size, n_heads, len_q, dim_v]; weights [batch_size, n_heads, len_q, len_k]
        context, weights = ScaleDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # 重塑上下文向量并进行线性变换
        # context [batch_size, len_q, n_heads * dim_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # output [batch_size, len_q, embedding_dim]
        output = self.linear(context)

        # 与输入(Q)进行残差链接，并进行层归一化后输出
        # output [batch_size, len_q, embedding_dim]
        output = self.layer_norm(output + residual)
        return output, weights  # 返回层归一化的输出和注意力权重


# 定义逐位置前向传播网络类
class PowiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PowiseFeedForwardNet, self).__init__()
        # 定义一维卷积层1，用于将输入映射到更高维度
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=2048, kernel_size=1)
        # 定义一维卷积层2，用于将输入映射回原始维度
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=d_embedding, kernel_size=1)
        # 定于归一化
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs):
        # inputs [batch_size, len_q, embedding_dim]
        residual = inputs
        # 在卷积层1后使用ReLU激活函数
        # output [batch_size, embedding_dim, len_q]->[batch_size, 2048, len_q]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # 使用卷积层2进行降维
        # output [batch_size, 2048, len_q]->[batch_size, len_q, embedding_dim]
        output = self.conv2(output).transpose(1, 2)
        # 与输入进行残差链接，并进行层归一化
        # output [batch_size, len_q, embedding_dim]
        output = self.layer_norm(output + residual)
        return output       # 返回加入残差连接后层归一化的结果


# 生成正弦位置编码表的函数，用于在Transformer中引入位置信息
def get_sin_enc_table(n_position, embedding_dim):
    # n_position：输入序列的最大长度
    # embedding_dim：词嵌入向量的维度

    # 根据位置和维度信息，初始化正弦位置编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))
    # 遍历所以位置和维度，计算角度值
    for i in range(n_position):
        for j in range(embedding_dim):
            angle = i / np.power(10000, 2 * (j // 2) / embedding_dim)
            sinusoid_table[i, j] = angle

    # 计算正弦和余弦值
    # sinusoid_table [n_position, embedding_dim]
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])   # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])   # dim 2i+1 奇数维
    return torch.FloatTensor(sinusoid_table)    # 返回正弦位置编码表


# 生成填充注意力掩码的函数，用于在多头自注意力计算中忽略填充部分
def get_attn_pad_mask(seq_q, seq_k):
    # seq_q [batch_size, len_q]
    # seq_k [batch_size, len_k]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # 生成bool类型的张量
    # pad_attn_mask [batch_size, 1, len_k(=len_q)]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # <PAD> Token的编码值为0
    # 变形为注意力分数相同形状的张量
    # pad_attm_mask [batch_size, len_q, len_k]
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    return pad_attn_mask


# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()   # 多头自注意力层
        self.pos_ffn = PowiseFeedForwardNet()       # 位置前馈神经网络层

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs [batch_size, seq_len, embedding_dim]
        # enc_self_attn_mask [batch_size, seq_len, seq_len]

        # 将相同的Q,K,V输入多头自注意力层
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs,enc_inputs,
                                                       enc_self_attn_mask)
        # 将多头自注意力outputs输入位置前馈神经网络层
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs [batch_size, seq_len, embedding_dim]
        # attn_weights [batch_size, n_heads, seq_len, seq_len]
        return enc_outputs, attn_weights    # 返回编码器输出和每层编码器注意力权重


#定义编码器类
n_layers = 6    # 设置Encoder/Decoder的层数
class Encoder(nn.Module):
    def __init__(self, corpus):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding)  # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(corpus.src_len+1, d_embedding), freeze=True)  # 位置嵌入层
        self.layers = nn.ModuleList(EncoderLayer() for i in range(n_layers))    # 编码器层数

    def forward(self, enc_inputs):
        # enc_inputs [batch_size, source_len]

        # 创建一个从1到source_len的位置索引序列
        # pos_indices [1, source_len]
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        # 对输入进行词嵌入和位置嵌入相加
        # enc_outputs [batch_size, source_len, embedding_dim]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        # enc_self_attn_mask [batch_size, len_q, len_k]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # 生成自注意力掩码

        enc_self_attn_weights = []
        # 通过编码器层
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)

        # enc_outputs [batch_size, seq_len, embedding_dim]
        # enc_self_attn_weights 是一个列表，每个元素的维度是[batch_size, n_heads, seq_len, seq_len]
        return enc_outputs, enc_self_attn_weights   # 返回编码器输出和编码器注意力权重


# 生成后续注意力掩码的函数，用于在多头自注意力计算中忽略未来信息
def get_attn_subsequent_mask(seq):
    # seq [batch_size, seq_len(Q)=seq_len(K)]

    # attn_shape是一个一维张量 [batch_size, seq_len(Q), seq_len(K)]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]    # 获取输入序列的形状
    # 使用numpy创建一个上三角矩阵（triu = triangle upper）
    # subsequent_mask [batch_size, seq_len(Q), seq_len(K)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    # 将numpy数组转换为PyTorch张量，并将数据类型设置为byte（布尔值）
    # subsequent_mask [batch_size, seq_len(Q), seq_len(K)]
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask  # 返回后续位置的注意力掩码


# 定义解码器层类
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()   # 多头自注意力层
        self.dec_enc_attn = MultiHeadAttention()    # 多头自注意力层，连接编码器和解码器
        self.pos_ffn = PowiseFeedForwardNet()   # 位置前馈神经网络层

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_inputs [batch_size, target_len, embedding_dim]
        # enc_outputs [batch_size, source_len, embedding_dim]
        # dec_self_attn_mask [batch_size, target_len, target_len]
        # dec_enc_attn_mask [batch_size, target_len, source_len]

        #将相同的K，Q，V输入多头自注意力层
        # dec_outputs [batch_size, target_len, embedding_dim]
        # dec_self_attn [batch_size, n_heads, target_len, target_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                       dec_self_attn_mask)
        # 将解码器输出和编码器输出输入多头注意力层
        # dec_outputs [batch_size, target_len, embedding_dim]
        # dec_enc_attn [batch_size, n_heads, target_len, source_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_enc_attn_mask)
        # 输入位置前馈神经网络层
        # dec_outputs [batch_size, target_len, embedding_dim]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn   # 返回编码器层输出，每层的自注意力和解-编码器注意力权重


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, corpus):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab), d_embedding)  # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(corpus.tgt_len+1, d_embedding), freeze=True)  # 位置嵌入层
        self.layers = nn.ModuleList([DecoderLayer() for i in range(n_layers)])  # 叠加多层

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # dec_inputs [batch_size, target_len]
        # enc_inputs [batch_size, source_len]
        # enc_outputs [batch_size, source_len, embedding_dim]

        # 创建一个从1到source_len的位置索引序列
        # pos_indices [1, target_len]
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        # 对输入进行词嵌入和位置嵌入相加
        # dec_outputs [batch_size, target_len, embedding_dim]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)

        # 生成解码器自注意力掩码和解码器-编码器注意力掩码
        # dec_self_attn_pad_mask [batch_size, target_len, target_len]
        # dec_self_attn_subsequent_mask [batch_size, target_len, target_len]
        # dec_self_attn_mask [batch_size, target_len, target_len]
        # dec_enc_attn_mask [batch_size, target_len, source_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # 填充位掩码
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)    # 后续位掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_subsequent_mask.to(device) +
                                      dec_self_attn_subsequent_mask.to(device)), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)   # 解码器-编码器掩码

        dec_self_attns, dec_enc_attns = [], []
        # 通过解码器层
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, target_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, source_len]
        return dec_outputs, dec_self_attns, dec_enc_attns # 返回解码器输出，解码器自注意力和解-编编码器注意力权重


# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus)  # 初始化编码器
        self.decoder = Decoder(corpus)  # 初始化解码器
        # 定义线性投影层，将解码器输出转换为目标词汇表大小的概率分布
        self.projection = nn.Linear(d_embedding, len(corpus.tgt_vocab), bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs [batch_size, source_seq_len]
        # dec_inputs [batch_size, target_seq_len]
        # 将输入传递给编码器，并获取编码器输出和自注意力权重
        # enc_outputs [batch_size, source_len, embedding_dim]
        # enc_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, src_seq_len, src_seq_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # 将编码器输出、解码器输入和编码器输入传递给解码器
        # 获取解码器输出、解码器自注意力权重和编码器-解码器注意力权重
        # dec_outputs [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # 将解码器输出传递给投影层，生成目标词汇表大小的概率分布
        # dec_logits [batch_size, tgt_seq_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        # 返回逻辑值(原始预测结果)，编码器自注意力权重，解码器自注意力权重，解-编码器注意力权重
        return dec_logits, enc_self_attns, dec_self_attns,dec_enc_attns