import os
from loguru import logger
import torch
from tqdm import trange, tqdm
import numpy as np
import pickle
from utils.utils import write_pickle, load_pickle
from utils.utils import load_lines, write_lines
from processors.trie_tree import Trie
from processors.dataset import NERDataset
import json
from processors.vocab import Vocabulary
from os.path import join
import argparse
from transformers import BertTokenizer


class Processor(object):

    def __init__(self, config):
        self.data_path = config.data_path
        self.overwrite = config.overwrite
        self.train_file = config.train_file
        self.dev_file = config.dev_file
        self.test_file = config.test_file

    def get_input_data(self, file):
        raise NotImplemented()

    def get_train_data(self):
        logger.info('loading train data')
        if type(self) == BertProcessor:
            file_name = 'bert_train.pkl'
        else:
            file_name = 'lebert_train.pkl'
        save_path = join(self.data_path, file_name)  #'datasets/msra/lebert_train.pkl'
        if self.overwrite or not os.path.exists(save_path):
            features = self.get_input_data(self.train_file)
            write_pickle(features, save_path)
        else:
            features = load_pickle(save_path)
        train_dataset = NERDataset(features)# 用特征建立NER数据集对象
        logger.info('len of train data:{}'.format(len(features)))
        return train_dataset

    def get_dev_data(self):
        logger.info('loading dev data')
        if type(self) == BertProcessor:
            file_name = 'bert_dev.pkl'
        else:
            file_name = 'lebert_dev.pkl'
        save_path = join(self.data_path, file_name)
        if self.overwrite or not os.path.exists(save_path):
            features = self.get_input_data(self.dev_file)
            write_pickle(features, save_path)
        else:
            features = load_pickle(save_path)
        dev_dataset = NERDataset(features)
        logger.info('len of dev data:{}'.format(len(features)))
        return dev_dataset

    def get_test_data(self):
        logger.info('loading test data')
        if type(self) == BertProcessor:
            file_name = 'bert_test.pkl'
        else:
            file_name = 'lebert_test.pkl'
        save_path = join(self.data_path, file_name)
        if self.overwrite or not os.path.exists(save_path):
            features = self.get_input_data(self.test_file)
            write_pickle(features, save_path)
        else:
            features = load_pickle(save_path)
        test_dataset = NERDataset(features)
        logger.info('len of test data:{}'.format(len(features)))
        return test_dataset


class LEBertProcessor(Processor):
    def __init__(self, args, tokenizer):
        super(LEBertProcessor, self).__init__(args)
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.data_path = args.data_path  
        self.max_seq_len = args.max_seq_len    #128
        self.max_word_num = args.max_word_num  #3
        self.overwrite = args.overwrite         #True
        self.output_path = args.output_path
        self.tokenizer = tokenizer
        data_files = [self.train_file, self.dev_file, self.test_file]
        self.word_embedding, self.word_vocab, self.label_vocab, self.trie_tree = self.init(
            args.pretrain_embed_path, args.output_path,  args.max_scan_num, data_files, args.label_path, args.overwrite  #args.max_scan_num=3000000取预训练词向量的前max_scan_num个构造字典树
        )#args.pretrain_embed_path='/root/autodl-fs/imcs21/task/NER/LEBERT-NER/Downloads/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'

    # def load_words_from_pretrain(self, word_embedding_path, save_path, overwrite):
    #     """
    #     获取预训练词表中的所有单词
    #     :param word_embedding_path:
    #     :param words_output_path
    #     :param overwrite
    #     :return:
    #     """
    #     logger.info('loading words from pretrain vocabulary')
    #     if overwrite or not os.path.exists(save_path):
    #         word_list = []
    #         with open(word_embedding_path, 'r', encoding='utf8') as reader, \
    #                 open(save_path, 'w', encoding='utf8') as writer:
    #             for word in enumerate(reader):
    #                 word = word.strip()
    #                 word_list.append(word)
    #                 writer.write('{}\n'.format(word))
    #     else:
    #         with open(save_path, 'r', encoding='utf8') as f:
    #             word_list = [word.strip() for word in enumerate(f)]
    #     logger.info('size of word list:{}'.format(len(word_list)))
    #     return word_list

    def init(self, pretrain_embed_path, output_path,  max_scan_num, data_files, label_path, overwrite):
        word_embed_path = join(self.data_path, 'word_embedding.pkl') #'datasets/msra/word_embedding.pkl'
        word_vocab_path = join(self.data_path, 'word_vocab.pkl') #'datasets/msra/word_vocab.pkl'
        word_vocab_path_ = join(self.data_path, 'word_vocab.txt')
        trie_tree_path = join(self.data_path, 'trie_tree.pkl')

        if overwrite or not os.path.exists(word_embed_path) or not os.path.exists(word_vocab_path):
            # 加载词向量
            word_embed_dict, word_list, word_embed_dim = self.load_word_embedding(pretrain_embed_path, max_scan_num) #调用self.load_word_embedding
            # 构建字典树
            trie_tree = self.build_trie_tree(word_list, trie_tree_path) # 保存到trie_tree_path
            # 找到数据集中的所有单词
            corpus_words = self.get_words_from_corpus(data_files, word_vocab_path_, trie_tree)
            # 初始化模型的词向量
            model_word_embedding, word_vocab, embed_dim = \
                self.init_model_word_embedding(corpus_words, word_embed_dict, word_embed_path, word_vocab_path)
        else:
            model_word_embedding = load_pickle(word_embed_path)
            word_vocab = load_pickle(word_vocab_path)
            trie_tree = load_pickle(trie_tree_path)
        # 加载label
        labels = load_lines(label_path) #['O', 'B-Drug', 'B-Drug_Category', 'B-Medical_Examination', 'B-Operation', 'B-Symptom', 'I-Drug', 'I-Drug_Category', 'I-Medical_Examination', 'I-Operation', 'I-Symptom']
        label_vocab = Vocabulary(labels, vocab_type='label')#id2token列表 token2id字典 和长度的对象
        return model_word_embedding, word_vocab, label_vocab, trie_tree

    @classmethod
    def load_word_embedding(cls, word_embed_path, max_scan_num):
        """
        todo 存在许多单字的，考虑是否去掉
        加载前args.max_scan_num=3000000个词向量, 并且返回词表
        :return:
        """
        logger.info('loading word embedding from pretrain')
        word_embed_dict = dict()
        word_list = list()
        with open(word_embed_path, 'r', encoding='utf8') as f:
            for idx, line in tqdm(enumerate(f)):
                # 只扫描前args.max_scan_num=3000000个词向量
                if idx > max_scan_num:
                    break
                items = line.strip().split() #  idx=0 line='2000000 200\n'   items=['2000000', '200']  #idx=1 items=['</s>', '0.002001', '0.002210', '-0.001915', '-0.001639', '0.000683', '0.001511', '0.000470', '0.000106', '-0.001802', '0.001109', '-0.002178', '0.000625', '-0.000376', ...]
                if idx == 0:
                    assert len(items) == 2
                    num_embed, word_embed_dim = items
                    num_embed, word_embed_dim = int(num_embed), int(word_embed_dim)#num_embed=2000000 word_embed_dim=200
                else:
                    assert len(items) == word_embed_dim + 1
                    word = items[0]               #第0个时单词 后面是向量
                    embedding = np.empty([1, word_embed_dim])
                    embedding[:] = items[1:]
                    word_embed_dict[word] = embedding
                    word_list.append(word)
        logger.info('word_embed_dim:{}'.format(word_embed_dim))
        logger.info('size of word_embed_dict:{}'.format(len(word_embed_dict)))
        logger.info('size of word_list:{}'.format(len(word_list)))

        return word_embed_dict, word_list, word_embed_dim

    @classmethod
    def build_trie_tree(cls, word_list, save_path):
        """
        # todo 是否不将单字加入字典树中  后面 use_single=True，应该是加入了
        构建字典树
        :return:
        """
        logger.info('building trie tree')
        trie_tree = Trie()              #/root/autodl-fs/imcs21/task/NER/LEBERT-NER/processors/trie_tree.py
        for word in word_list:
            trie_tree.insert(word) #向字典树中添加
        write_pickle(trie_tree, save_path) #/root/autodl-fs/imcs21/task/NER/LEBERT-NER/utils/utils.py   def write_pickle(x, path):with open(path, 'wb') as f:pickle.dump(x, f) 保存，Json不支持自定义类
        return trie_tree #保存完对象，再返回对象

    @classmethod
    def get_words_from_corpus(cls, files, save_file, trie_tree):
        """
        找出文件中所有匹配的单词
        :param files:
        :return:
        """
        logger.info('getting words from corpus')
        all_matched_words = set()
        for file in files: #['datasets/msra/train.json', 'datasets/msra/dev.json', 'datasets/msra/test.json']
            with open(file, 'r', encoding='utf8') as f:
                lines = f.readlines()   #读取所有行到列表 第一行：'{"text": ["医", "生", "：", "你", "好"], "label": ["O", "O", "O", "O", "O"]}\n'
                for idx in trange(len(lines)):
                    line = lines[idx].strip() #去掉末尾的换行符
                    data = json.loads(line)#把字符串当做json读取成字典 {'text': ['医', '生', '：', '你', '好'], 'label': ['O', 'O', 'O', 'O', 'O']}
                    text = data['text']
                    matched_words = cls.get_words_from_text(text, trie_tree)  # 处理text，找出所有匹配的单词
                    _ = [all_matched_words.add(word) for word in matched_words]

        all_matched_words = list(all_matched_words)
        all_matched_words = sorted(all_matched_words)
        write_lines(all_matched_words, save_file)
        return all_matched_words

    @classmethod
    def get_words_from_text(cls, text, trie_tree):
        """
        找出text中所有的单词
        :param text:
        :param trie_tree:
        :return:
        """
        length = len(text)
        matched_words_set = set()   # 存储匹配到的单词
        for idx in range(length):
            sub_text = text[idx:idx + trie_tree.max_depth] #trie_tree.max_depth=30
            words = trie_tree.enumerateMatch(sub_text)

            _ = [matched_words_set.add(word) for word in words]
        matched_words_set = list(matched_words_set)
        matched_words_set = sorted(matched_words_set)
        return matched_words_set

    def init_model_word_embedding(self, corpus_words, word_embed_dict, save_embed_path, save_word_vocab_path):
        logger.info('initializing model word embedding')
        # 构建数据集中单词和id的映射
        word_vocab = Vocabulary(corpus_words, vocab_type='word')  #/root/autodl-fs/imcs21/task/NER/LEBERT-NER/processors/vocab.py
        # embed_dim = len(word_embed_dict.items()[1].size)
        embed_dim = next(iter(word_embed_dict.values())).size  #200

        scale = np.sqrt(3.0 / embed_dim)
        model_word_embedding = np.empty([word_vocab.size, embed_dim])

        matched = 0
        not_matched = 0

        for idx, word in enumerate(word_vocab.idx2token):
            if word in word_embed_dict:  #如果单词出现在预训练词向量中，加载词向量
                model_word_embedding[idx, :] = word_embed_dict[word]
                matched += 1
            else: #如果单词不在预训练词向量中，随机初始化
                model_word_embedding[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
                not_matched += 1

        logger.info('num of match:{}, num of not_match:{}'.format(matched, not_matched))
        write_pickle(model_word_embedding, save_embed_path)#数据集的词向量，
        write_pickle(word_vocab, save_word_vocab_path)#

        return model_word_embedding, word_vocab, embed_dim

    def get_char2words(self, text):
        """
        获取每个汉字，对应的单词列表
        :param text:
        :return:
        """
        text_len = len(text) #['医', '生', '：', '在', '吗']
        char_index2words = [[] for _ in range(text_len)]  #[[], [], [], [], []]

        for idx in range(text_len):
            sub_sent = text[idx:idx + self.trie_tree.max_depth]  # speed using max depth
            words = self.trie_tree.enumerateMatch(sub_sent)  # 找到以text[idx]开头的所有单词
            for word in words:
                start_pos = idx
                end_pos = idx + len(word)
                for i in range(start_pos, end_pos):
                    char_index2words[i].append(word)
        # todo 截断
        # for i, words in enumerate(char_index2words):
        #     char_index2words[i] = char_index2words[i][:self.max_word_num]
        return char_index2words

    def get_input_data(self, file):
        lines = load_lines(file)
        features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        o_label_id = self.label_vocab.convert_token_to_id('O')
        pad_label_id = self.label_vocab.convert_token_to_id('[PAD]')

        for line in tqdm(lines):
            data = json.loads(line) #{'text': ['医', '生', '：', '你', '好'], 'label': ['O', 'O', 'O', 'O', 'O']}
            text = data['text']
            labels = data['label']
            char_index2words = self.get_char2words(text) #[['医生'], ['医生', '生'], ['：'], ['你好'], ['你好', '好']]

            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text) + [sep_token_id]
            label_ids = [o_label_id] + self.label_vocab.convert_tokens_to_ids(labels) + [o_label_id]

            word_ids_list = []
            word_pad_id = self.word_vocab.convert_token_to_id('[PAD]')#0
            for words in char_index2words:
                words = words[:self.max_word_num]   #self.max_word_num=3  不够3个的补0
                word_ids = self.word_vocab.convert_tokens_to_ids(words)
                word_pad_num = self.max_word_num - len(words)
                word_ids = word_ids + [word_pad_id] * word_pad_num
                word_ids_list.append(word_ids) #[[15547, 0, 0], [15547, 43389, 0], [59349, 0, 0], [10721, 0, 0], [10721, 24111, 0]]
            # 开头和结尾进行padding
            word_ids_list = [[word_pad_id]*self.max_word_num] + word_ids_list + [[word_pad_id]*self.max_word_num]

            if len(input_ids) > self.max_seq_len:  #input_ids=[101, 1278, 4495, 8038, 872, 1962, 102]   self.max_seq_len=128 
                input_ids = input_ids[: self.max_seq_len]
                label_ids = label_ids[: self.max_seq_len]
                word_ids_list = word_ids_list[: self.max_seq_len]
            input_mask = [1] * len(input_ids)  #[1, 1, 1, 1, 1, 1, 1]
            token_type_ids = [0] * len(input_ids)
            assert len(input_ids) == len(label_ids) == len(word_ids_list)

            # padding
            padding_length = self.max_seq_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length #[101, 1278, 4495, 8038, 872, 1962, 102, 0, 0, 0, 0, 0, 0, 0, ...]
            input_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            label_ids += [pad_label_id] * padding_length #[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...]
            word_ids_list += [[word_pad_id]*self.max_word_num] * padding_length  #补3个0  [[0, 0, 0], [15547, 0, 0], [15547, 43389, 0], [59349, 0, 0], [10721, 0, 0], ……

            text = ''.join(text)#去掉空格  
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            token_type_ids = torch.LongTensor(token_type_ids)
            word_ids = torch.LongTensor(word_ids_list)
            word_mask = (word_ids != word_pad_id).long()

            feature = {
                'text': text, 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids,
                'word_ids': word_ids, 'word_mask': word_mask, 'label_ids': label_ids
            }
            features.append(feature)

        return features

    # def get_train_data(self):
    #     logger.info('loading train data')
    #     save_path = join(self.data_path, 'train.pkl')
    #     if self.overwrite or not os.path.exists(save_path):
    #         features = self.get_input_data(self.train_file)
    #         write_pickle(features, save_path)
    #     else:
    #         features = load_pickle(save_path)
    #     train_dataset = NERDataset(features)
    #     logger.info('len of train data:{}'.format(len(features)))
    #     return train_dataset
    #
    # def get_dev_data(self):
    #     logger.info('loading dev data')
    #     save_path = join(self.data_path, 'dev.pkl')
    #     if self.overwrite or not os.path.exists(save_path):
    #         features = self.get_input_data(self.dev_file)
    #         write_pickle(features, save_path)
    #     else:
    #         features = load_pickle(save_path)
    #     dev_dataset = NERDataset(features)
    #     logger.info('len of dev data:{}'.format(len(features)))
    #     return dev_dataset
    #
    # def get_test_data(self):
    #     logger.info('loading test data')
    #     save_path = join(self.data_path, 'test.pkl')
    #     if self.overwrite or not os.path.exists(save_path):
    #         features = self.get_input_data(self.test_file)
    #         write_pickle(features, save_path)
    #     else:
    #         features = load_pickle(save_path)
    #     test_dataset = NERDataset(features)
    #     logger.info('len of test data:{}'.format(len(features)))
    #     return test_dataset


class BertProcessor(Processor):
    def __init__(self, args, tokenizer):
        super(BertProcessor, self).__init__(args)
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.data_path = args.data_path
        self.max_seq_len = args.max_seq_len
        self.max_word_num = args.max_word_num
        self.overwrite = args.overwrite
        self.output_path = args.output_path
        self.tokenizer = tokenizer

        # 加载label
        labels = load_lines(args.label_path)
        self.label_vocab = Vocabulary(labels, vocab_type='label')

    def get_input_data(self, file):
        lines = load_lines(file)
        features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        o_label_id = self.label_vocab.convert_token_to_id('O')
        pad_label_id = self.label_vocab.convert_token_to_id('[PAD]')

        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            labels = data['label']

            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text) + [sep_token_id]
            label_ids = [o_label_id] + self.label_vocab.convert_tokens_to_ids(labels) + [o_label_id]

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                label_ids = label_ids[: self.max_seq_len]
            input_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            assert len(input_ids) == len(label_ids)

            # padding
            padding_length = self.max_seq_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length
            input_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            label_ids += [pad_label_id] * padding_length

            text = ''.join(text)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            token_type_ids = torch.LongTensor(token_type_ids)

            feature = {
                'text': text, 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids,
                'label_ids': label_ids
            }
            features.append(feature)

        return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_file = '../datasets/resume/train.json'
    args.dev_file = '../datasets/resume/dev.json'
    args.test_file = '../datasets/resume/test.json'
    args.max_seq_len = 150
    args.max_word_num = 5
    args.pretrain_embed_path = '/Users/yangjianxin/Downloads/tencent-ailab-embedding-zh-d200-v0.2.0/tencent-ailab-embedding-zh-d200-v0.2.0.txt'
    args.output_path = '../output/cner'
    args.max_scan_num = 10000
    args.label_path = '../datasets/resume/labels.txt'
    args.data_path = '../datasets/resume'
    args.overwrite = True
    tokenizer = BertTokenizer.from_pretrained('../pretrain_model/bert-base-chinese')
    # processor = Processor(args, tokenizer)
    # train_set = processor.get_dev_data()
    processor = BertProcessor(args, tokenizer)
    train_set = processor.get_dev_data()
    #
    # with open(word_embedding_path, 'r', encoding='utf8') as f:
    #     for idx, line in tqdm(enumerate(f)):
    #         # 只扫描前max_scan_num个词向量
    #         if idx > 1000000:
    #             break
    #         items = line.strip().split()

class CnerProcessor():
    """Processor for the chinese ner data set."""
    def __init__(self, train_path, dev_path, test_path, tokenizer, max_len, segment_a_id=0):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.segment_a_id = segment_a_id
        self.label2id = {label: i for i, label in enumerate(self.get_labels())}  # 每种标签对应的id
        self.id2label = {i: label for i, label in enumerate(self.get_labels())}  # 每种标签对应的id

    def get_train_examples(self):
        examples = self.create_examples(self.read_text(self.train_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of train data:{}'.format(len(features)))
        return features

    def get_dev_examples(self):
        examples = self.create_examples(self.read_text(self.dev_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of dev data:{}'.format(len(features)))
        return features

    def get_test_examples(self):
        examples = self.create_examples(self.read_text(self.test_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of test data:{}'.format(len(features)))
        return features

    def convert_examples_to_inputs(self, examples):
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        o_label_id = self.label2id['O']
        features = []
        for words, labels in examples:
            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(words) + [sep_token_id]
            label_ids = [o_label_id] + [self.label2id[x] for x in labels] + [o_label_id]
            if len(input_ids) > self.max_len:
                input_ids = input_ids[: self.max_len]
                label_ids = label_ids[: self.max_len]
            input_mask = [1] * len(input_ids)
            token_type_ids = [self.segment_a_id] * len(input_ids)
            assert len(input_ids) == len(label_ids)

            # 对输入进行padding
            padding_length = self.max_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length
            input_mask += [pad_token_id] * padding_length
            token_type_ids += [pad_token_id] * padding_length
            label_ids += [pad_token_id] * padding_length
            text = ''.join(words)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            token_type_ids = torch.LongTensor(token_type_ids)
            feature = {'text': text, 'input_ids': input_ids, 'label_ids': label_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
            features.append(feature)
        return features

    def read_text(self, file):
        """
        读取文件，将每条记录读取为words:['我','在','天','津']，labels:['O','O','B-ORG','I-ORG']
        :param file:
        :return:
        """
        lines = []
        with open(file, 'r', encoding='utf8') as f:
            words = []
            labels = []
            for line in f:
                if line == "" or line == "\n":
                    # 读取完一条记录
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    def get_labels(self):
        """See base class."""
        # return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
        #         'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
        #         'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]
        return ['B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'S-NAME', 'S-ORG', 'S-RACE', 'O']

    def create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            words = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append((words, labels))
        return examples
