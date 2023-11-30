from collections import Counter
import torch
from torch import nn
import torch.optim as optim
from Transformer import Transformer
import matplotlib.pyplot as plt # 导入matplotlib
from matplotlib.ticker import FixedLocator # # 导入FixedLocator
import seaborn as sns

class TranslationCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
        self.src_len = max(len(sentence[0].split()) for sentence in sentences) + 1
        self.tgt_len = max(len(sentence[1].split()) for sentence in sentences) + 2
        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}

    # 创建词汇表
    def create_vocabularies(self):
        src_counter = Counter(word for sentence in self.sentences for word in sentence[0].split())
        tgt_counter = Counter(word for sentence in self.sentences for word in sentence[1].split())
        src_vocab = {'<pad>': 0, **{word: i+1 for i, word in enumerate(src_counter)}}
        tgt_vocab = {'<pad>': 0, '<sos>': 1,'<eos>': 2,
                     **{word: i+3 for i, word in enumerate(tgt_counter)}}
        return src_vocab,tgt_vocab

    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch, target_batch = [], [], []
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        for i in sentence_indices:
            src_sentence, tgt_sentence = self.sentences[i]
            src_seq = [self.src_vocab[word] for word in src_sentence.split()]
            tgt_seq = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word] for word in tgt_sentence.split()]\
                + [self.tgt_vocab['<eos>']]
            src_seq += [self.src_vocab['<pad>']] * (self.src_len - len(src_seq))
            tgt_seq += [self.tgt_vocab['<pad>']] * (self.tgt_len - len(tgt_seq))
            input_batch.append(src_seq)
            output_batch.append([self.tgt_vocab['<sos>']] + ([self.tgt_vocab['<pad>']] *
                                (self.tgt_len - 2)) if test_batch else tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])

        input_batch = torch.LongTensor(input_batch)
        output_batch = torch.LongTensor(output_batch)
        target_batch = torch.LongTensor(target_batch)
        return input_batch, output_batch, target_batch

def train_model(model, corpus):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for i in range(100):
        optimizer.zero_grad()
        enc_inputs, dec_inputs, target_batch = corpus.make_batch(10)
        enc_inputs, dec_inputs, target_batch = enc_inputs.to(device), \
                                               dec_inputs.to(device), target_batch.to(device)
        outputs, _, _, _ = model(enc_inputs, dec_inputs)
        loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1))
        if (i + 1) % 20 == 0:
            print(f"Epoch：{i + 1:04d}, cost = {loss:.6f}")
        loss.backward()
        optimizer.step()

def showgraph1(attn, n_heads, src_input, tgt_input, attention_type='enc'):
    plt.rcParams["font.family"] = ['SimHei']  # 用来设定字体样式
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设定无衬线字体样式
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    attn = attn[-1].squeeze(0)[0]  # 提取最后一层注意力权重的第一个头
    attn = attn.squeeze(0).cpu().data.numpy()  # 去除冗余维度并转换为numpy数组
    fig = plt.figure(figsize=(n_heads, n_heads))  # 创建一个新的图形
    ax = fig.add_subplot(1, 1, 1)  # 添加一个子图
    ax.matshow(attn, cmap='viridis')  # 绘制注意力权重矩阵
    if attention_type == 'enc':
        src_labels = [''] + [corpus.src_idx2word[idx.item()] for idx in src_input]
        tgt_labels = src_labels
    elif attention_type == 'dec':
        src_labels = [''] + [corpus.tgt_idx2word[idx.item()] for idx in tgt_input]
        tgt_labels = src_labels
    elif attention_type == 'dec_enc':
        src_labels = [''] + [corpus.src_idx2word[idx.item()] for idx in src_input]
        tgt_labels = [''] + [corpus.tgt_idx2word[idx.item()] for idx in tgt_input]
    ax.xaxis.set_major_locator(FixedLocator(range(len(src_labels))))  # 设置FixedLocator
    ax.yaxis.set_major_locator(FixedLocator(range(len(tgt_labels))))  # 设置FixedLocator
    ax.set_xticklabels(src_labels, fontdict={'fontsize': 14}, rotation=90)  # 设置x轴标签
    ax.set_yticklabels(tgt_labels, fontdict={'fontsize': 14})  # 设置y轴标签
    plt.show()  # 显示图形

def showgraph(attn, n_heads, src_input, tgt_input, head_idx=0):
    plt.rcParams["font.family"] = ['SimHei']  # 用来设定字体样式
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设定无衬线字体样式
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 选择一个特定的头并删除额外的维度
    attn = attn[-1].squeeze(0)[head_idx].cpu().data.numpy()
    plt.figure(figsize=(n_heads, n_heads)) # 创建一个新的图形
    # 将输入索引转换为单词标签
    src_labels = [corpus.src_idx2word[idx.item()] for idx in src_input]
    tgt_labels = [corpus.tgt_idx2word[idx.item()] for idx in tgt_input]
    # 使用seaborn库绘制热力图
    sns.heatmap(attn, cmap='viridis',
                xticklabels=src_labels,
                yticklabels=tgt_labels,
                annot=True)
    plt.show()# 显示图形

def test_model(model, corpus):
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True)
    enc_inputs, dec_inputs, target_batch = enc_inputs.to(device), \
                                           dec_inputs.to(device), target_batch.to(device)
    print("编码器输入：", enc_inputs)
    print("解码器输入：", dec_inputs)
    print("目标数据：", target_batch)
    predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    predict = predict.view(-1, len(corpus.tgt_vocab))
    predict = predict.data.max(1, keepdim=True)[1]
    print("翻译结果：", predict)
    translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
    input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])
    print(input_sentence, '->', translated_sentence)

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns, 8, enc_inputs[0], enc_inputs[0])  # 显示编码器自注意力权重
    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns, 8, dec_inputs[0], dec_inputs[0])  # 显示解码器自注意力权重
    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns, 8, enc_inputs[0], dec_inputs[0])  # 解码器-编码器注意力权重

if __name__=='__main__':
    sentences = [
        ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
        ['我 爱 学习 人工智能', 'I love studying AI'],
        ['深度学习 改变 世界', 'Deep learning YYDS'],
        ['自然语言处理 很 强大', 'NLP is powerful'],
        ['神经网络 非常 复杂', 'Neural-networks are complex']
    ]
    corpus = TranslationCorpus(sentences)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(corpus).to(device)
    print(corpus.src_vocab)
    print(corpus.tgt_vocab)

    train_model(model, corpus)
    test_model(model, corpus)