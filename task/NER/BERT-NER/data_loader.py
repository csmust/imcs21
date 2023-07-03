import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_seq_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    基础数据样本实例
    """

    def __init__(self, guid, words, seq_labels=None):
        self.guid = guid
        self.words = words
        self.seq_labels = seq_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """深拷贝"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """将实例序列化为json"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"  # indent是缩进打印


class InputFeatures(object):
    """数据集特征"""

    def __init__(self, input_ids, attention_mask, token_type_ids, seq_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.seq_labels_ids = seq_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """深拷贝"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """将实例序列化为json"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    """数据集处理"""

    def __init__(self, args):
        self.args = args
        self.seq_labels = get_seq_labels(args)  # 获得序列标签  一共多少种标签的列表

        self.input_text_file = 'input.seq.char'  # 文本文件
        self.seq_labels_file = 'output.seq.bio'  # 标签文件

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """文件读取"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, seqs, set_type):
        """特征转化  set_type = train/valid/"""
        examples = []
        for i, (text, seq) in enumerate(zip(texts, seqs)):
            guid = "%s-%s" % (set_type, i)
            words = text.split()  # 以空格分割 单词
            seq_labels = []
            for s in seq.split():
                seq_labels.append(self.seq_labels.index(s) if s in self.seq_labels else self.seq_labels.index("UNK"))

            assert len(words) == len(seq_labels)
            examples.append(InputExample(guid=guid, words=words, seq_labels=seq_labels))
        return examples

    def get_examples(self, mode):
        """
        获得训练/验证的内容
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),                  # 读取文本文件
                                     seqs=self._read_file(os.path.join(data_path, self.seq_labels_file)),                      # 读取标签文件
                                     set_type=mode)                                                                          # train/valid/test 模式


# 自定义任务处理器
processors = {
    "sample": Processor,
    "ner_data": Processor
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,    # 好像train dev 都是0传入
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """将InputExample对象转化为输入特征"""
    # 设置一些常用的符号
    cls_token = tokenizer.cls_token        # [CLS]
    sep_token = tokenizer.sep_token        # [SEP]
    unk_token = tokenizer.unk_token        # [UNK]
    pad_token_id = tokenizer.pad_token_id  # 0

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = []
        seq_labels_ids = []
        for word, seq_label in zip(example.words, example.seq_labels):
            word_tokens = tokenizer.tokenize(word)    # 一个字一个字的读，一个字一个字的分词！！！！！！！！！！！！！！！！！
            if not word_tokens:
                word_tokens = [unk_token]  # 处理UNK的token
            if len(word_tokens) > 1:
                logger.info("Word split into subwords: %s" % " ".join(word_tokens))
            tokens.extend(word_tokens)      #   extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
            # 对于token的第一个标记使用真实的标签id，其余标记使用pad   ；   token可能被分成多个词，所以标签也要对应
            seq_labels_ids.extend([int(seq_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # 对于过长的序列，截取前一部分进行训练
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:   #由于tokens还没有[CLS]和[SEP]，所以要减去2
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            seq_labels_ids = seq_labels_ids[:(max_seq_len - special_tokens_count)]

        # 添加 [SEP]
        tokens += [sep_token]
        seq_labels_ids += [pad_token_label_id]   # [SEP]的标签为pad的id=0     这里面O的id为2 
        # 对于句子对任务，属于句子A的token为0，句子B的token为1；对于分类任务，只有一个输入句子，全为0  
        token_type_ids = [sequence_a_segment_id] * len(tokens)  #对于分类任务，只有一个输入句子，全为0

        # 添加 [CLS]
        tokens = [cls_token] + tokens
        seq_labels_ids = [pad_token_label_id] + seq_labels_ids   ## [CLS]的标签为pad的id=0     这里面O的id为2 
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # 获得token在词汇表中的索引！！！！！！！！！！！！！！！！
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 非填充部分的token对应1
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)  #attention_mask [1, 1, 1, 1, 1, 1, 1]  好像还没填充呢

        # pad  开始填充至最大长度
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)  #[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...]
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
        seq_labels_ids = seq_labels_ids + ([pad_token_label_id] * padding_length)  #[0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, ...]

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(seq_labels_ids) == max_seq_len, "Error with seq labels length {} vs {}".format(len(seq_labels_ids),
                                                                                                  max_seq_len)

        # 查看前5个样本
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("seq_labels: %s" % " ".join([str(x) for x in seq_labels_ids]))
            
        # # 将特征输出到txt文件中，方便查看
        # with open('snips_train.txt', 'a+', encoding='utf-8') as f:
        #     f.write('guid:' + example.guid + '\n')
        #     f.write('tokens:')
        #     f.write(''.join([str(x) + ' ' for x in tokens]) + '\n')
        #     f.write('input_ids:')
        #     f.write(''.join([str(x) + ' ' for x in input_ids]) + '\n')
        #     f.write('attention_mask:')
        #     f.write(''.join([str(x) + ' ' for x in attention_mask]) + '\n')
        #     f.write('token_type_ids:')
        #     f.write(''.join([str(x) + ' ' for x in token_type_ids]) + '\n')
        #     f.write('seq_labels:')
        #     f.write(''.join([str(x) + ' ' for x in seq_labels_ids]) + '\n')
        # f.close()

        # 返回固定格式的特征
        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          seq_labels_ids=seq_labels_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    """读取缓存的数据"""
    # 获取自定义任务处理器
    processor = processors[args.task](args)

    # 获取缓存文件路径
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )
    # 如果存在缓存数据就加载，反之则重新生成
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train") 
            '''examples[0] ={
  "guid": "train-0",
  "seq_labels": [
    2,
    2,
    2,
    2,
    2
  ],
  "words": [
    "医",
    "生",
    "：",
    "你",
    "好"
  ]
}
  '''
        elif mode == "dev":
            examples = processor.get_examples("dev")
        else:
            raise Exception("For mode, Only train, dev is available")

        # 转化为特征
        pad_token_label_id = args.ignore_index  #  pad_token_label_id = args.ignore_index   0
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # 转化为Tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_seq_labels_ids = torch.tensor([f.seq_labels_ids for f in features], dtype=torch.long)
    #通过TensorDataset将Tensors数据包装成dataset格式
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_seq_labels_ids)
    return dataset

