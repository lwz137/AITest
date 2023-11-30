import torch
import jieba
import re
from collections import Counter
from torch.utils.data import Dataset

# WikiCorpus语料库
class WikiCorpus:
    def __init__(self, sentences, max_seq_len=256):
        self.sentences = sentences
        self.seq_len = max_seq_len
        self.vocab = self.create_vocabularies()
        self.vocab_size = len(self.vocab)
        self.idx2word = {v: k for k, v in self.vocab.items()}

    def create_vocabularies(self):
        # counter = Counter(word for sentence in self.sentences for word in sentence.split())
        # vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, **{word: i+3 for i, word in enumerate(counter)}}
        with open("all_sentences.txt", "r") as f:
            vocab = {line.split()[0]: int(line.split()[1]) for line in f}
        return vocab

    def make_batch(self, batch_size):
        input_batch, target_batch = [], []

        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        for i in sentence_indices:
            sentence = self.sentences[i]
            words = sentence.split()[:self.seq_len - 2]
            seq = [self.vocab['<sos>']] + [self.vocab[word] for word in words] + [self.vocab['<eos>']]
            seq += [self.vocab['<pad>']] * (self.seq_len - len(seq))
            input_batch.append(seq[:-1])
            target_batch.append(seq[1:])

        input_batch = torch.LongTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)
        return input_batch, target_batch


class TranslationDataset(Dataset):
    def __init__(self, sentences, word2idx_cn, word2idx_en):
        self.sentences = sentences
        self.word2idx_cn = word2idx_cn
        self.word2idx_en = word2idx_en

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence_cn = [self.word2idx_cn[word] for word in self.sentences[index][0].split()]
        sentence_en = [self.word2idx_en[word] for word in self.sentences[index][1].split()]
        sentence_en_in = sentence_en[:-1]
        sentence_en_out = sentence_en[1:]
        return torch.tensor(sentence_cn), torch.tensor(sentence_en_in), torch.tensor(sentence_en_out)

class CorpusLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sentences = []
        self.word_list_cn = ['<pad>']
        self.word_list_en = ['<pad>', '<sos>', '<eos>']

    def process_sentences(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            all_sentences = f.readlines()

        for i in range(0, len(all_sentences), 2):
            sentence_cn = ' '.join(jieba.cut(all_sentences[i].strip(), cut_all=False))
            sentence_en = ' '.join(re.findall(r'\b\w+\b', all_sentences[i+1].strip()))
            self.sentences.append([sentence_cn, '<sos> ' + sentence_en + ' <eos>'])

    def build_vocab(self):
        for s in self.sentences:
            self.word_list_cn.extend(s[0].split())
            self.word_list_en.extend(s[1].split())

        self.word_list_cn = list(set(self.word_list_cn))
        self.word_list_en = list(set(self.word_list_en))
        for token in ['<pad>', '<sos>', '<eos>']:
            if token in self.word_list_en:
                self.word_list_en.remove(token)
            self.word_list_en.insert(0, token)

        self.word2idx_cn = {w: i for i, w in enumerate(self.word_list_cn)}
        self.word2idx_en = {w: i for i, w in enumerate(self.word_list_en)}
        self.idx2word_cn = {i: w for i, w in enumerate(self.word_list_cn)}
        self.idx2word_en = {i: w for i, w in enumerate(self.word_list_en)}

        self.src_vocab = len(self.word2idx_cn)
        self.tgt_vocab = len(self.word2idx_en)
        self.src_len = max(len(sentence.split()) for sentence, _ in self.sentences)
        self.tgt_len = max(len(sentence.split()) for _, sentence in self.sentences)

    def create_dataset(self):
        return TranslationDataset(self.sentences, self.word2idx_cn, self.word2idx_en)