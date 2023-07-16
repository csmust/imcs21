import json
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


data_dir = '../../../dataset'
# data_dir = 'dataset'
train_set = load_json(os.path.join(data_dir, 'train.json'))
dev_set = load_json(os.path.join(data_dir, 'dev.json'))
test_set = load_json(os.path.join(data_dir, 'test.json'))

os.makedirs('datasets/msra', exist_ok=True)


def make_label():
    labels = ['O', 'B-Drug', 'B-Drug_Category', 'B-Medical_Examination', 'B-Operation',
              'B-Symptom', 'I-Drug', 'I-Drug_Category', 'I-Medical_Examination', 'I-Operation', 'I-Symptom']
    with open('datasets/msra/labels.txt', 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')
def make_intent():
    '''
    '''
    intents =[ 'Request-Etiology','Request-Precautions','Request-Medical_Advice','Inform-Etiology',
              'Diagnose','Request-Basic_Information','Request-Drug_Recommendation','Inform-Medical_Advice',
              'Request-Existing_Examination_and_Treatment','Inform-Basic_Information','Inform-Precautions',
              'Inform-Existing_Examination_and_Treatment','Inform-Drug_Recommendation','Request-Symptom','Inform-Symptom',
              'Other']
    with open('datasets/msra/intents.txt', 'w', encoding='utf-8') as f:
        for intent in intents:
            f.write(intent + '\n')

context_size = 2
def make_data(samples, mode='train'):
    with open('datasets/msra/{}.json'.format(mode), 'w', encoding='utf-8') as f:
        for pid, sample in samples.items():
            context=[]
            for sent in sample['dialogue']:
                context=context[-context_size:]
                text = [w for w in sent['speaker'] + '：' + sent['sentence']]
                label = ['O'] * 3 + sent['BIO_label'].split()
                intent = sent['dialogue_act'].split()
                f.write(json.dumps({'text': text, 'label': label , 'intent' : intent ,'context' : context}, ensure_ascii=False) + '\n')
                context.append(text)


make_label()
make_intent()

make_data(train_set, mode='train')
make_data(dev_set, mode='dev')
make_data(test_set, mode='test')
