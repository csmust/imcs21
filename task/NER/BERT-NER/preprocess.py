import json
import os


def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data


def save_data(samples, input_fn, output_fn2):  #input_fn: 保存句子的文件名，output_fn2: 保存bio的文件名
    seq_in = []
    seq_bio = []
    for pid, sample in samples.items():
        for item in sample['dialogue']:
            sent = list(item['speaker'] + '：' + item['sentence'])  # speaker + sentence 同时转为字符列表
            bio = ['O'] * 3 + item['BIO_label'].split(' ')
            assert len(sent) == len(bio)
            seq_in.append(sent)
            seq_bio.append(bio)
    assert len(seq_in) == len(seq_bio)
    print('句子数量为：', len(seq_in))
    with open(input_fn, 'w', encoding='utf-8') as f1:
        for i in seq_in:
            tmp = ' '.join(i)# 以空格分隔 将列表转为字符串
            f1.write(tmp + '\n')
    with open(output_fn2, 'w', encoding='utf-8') as f2:
        for i in seq_bio:
            tmp = ' '.join(i)
            f2.write(tmp + '\n')


def get_vocab_bio(fr1, fr2, fw): #fr1=训练集的bio文件，fr2=验证集的bio文件，fw=保存bio信息的文件
    bio = []
    with open(fr1, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in bio:  #去重
                    bio.append(i)
    with open(fr2, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in bio: #去重
                    bio.append(i)
    #因此，排序的规则是先按照字符串的第三个索引位置开始到末尾的子串进行排序，如果两个字符串的这个子串相同，则按照字符串的前两个索引位置的子串进行排序。
    bio = sorted(list(bio), key=lambda x: (x[2:], x[:2]))  #sorted() 函数对所有可迭代的对象进行排序操作。
    add_tokens = ["PAD", "UNK"]
    bio = add_tokens + bio
    print('bio种类：', len(bio))

    with open(fw, 'w', encoding='utf-8') as f:
        for w in bio:
            f.write(w + '\n')


if __name__ == "__main__":

    data_dir = '../../../dataset'
    # data_dir = 'dataset'
    train_set = load_json(os.path.join(data_dir, 'train.json'))
    dev_set = load_json(os.path.join(data_dir, 'dev.json'))

    saved_dir = 'ner_data'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        os.makedirs(os.path.join(saved_dir, 'train'))
        os.makedirs(os.path.join(saved_dir, 'dev'))

    # 获得训练数据
    save_data(
        train_set,
        os.path.join(saved_dir, 'train', 'input.seq.char'),
        os.path.join(saved_dir, 'train', 'output.seq.bio')
    )

    # 获得验证数据
    save_data(
        dev_set,
        os.path.join(saved_dir, 'dev', 'input.seq.char'),
        os.path.join(saved_dir, 'dev', 'output.seq.bio')
    )

    # 获取一些vocab信息
    get_vocab_bio(
        os.path.join(saved_dir, 'train', 'output.seq.bio'),
        os.path.join(saved_dir, 'dev', 'output.seq.bio'),
        os.path.join(saved_dir, 'vocab_bio.txt')
    )
