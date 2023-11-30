from collections import Counter

if __name__=='__main__':
    sentences = [
        ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
        ['我 爱 学习 人工智能', 'I love studying AI'],
        ['深度学习 改变 世界', 'Deep learning YYDS'],
        ['自然语言处理 很 强大', 'NLP is powerful'],
        ['神经网络 非常 复杂', 'Neural-networks are complex']
    ]
    src_counter = Counter(word for sentence in sentences for word in sentence[0].split())
    tgt_counter = Counter(word for sentence in sentences for word in sentence[1].split())
    # print(src_counter)
    # print(tgt_counter)
    src_vocab = {'<pad>': 0, **{word: i + 1 for i, word in enumerate(src_counter)}}
    tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2,
                     **{word: i + 3 for i, word in enumerate(tgt_counter)}}
    # print(src_vocab)
    # print(tgt_vocab)