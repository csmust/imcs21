import re
import torch


def is_slot_da(da):
    '''
    区分四元组是slot标注出来的还是intent分类出来的
    '''
    # if j['value'] is not None and j['value'] != '' and j['value'] != '？' and j['value'] != '?':
    if da[3] != '':
        # print(da[3])
        # 如果value有值就是tag标注
        return True
    return False


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    accnum=0
    TP_pre=0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        # print("预测：\t",predicts)
        # print("标签：\t",labels)
        # print("---"*30)
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                #错了
                FP += 1
                if item.get("ori"):
                    if item.get("FP") is None:
                        item["FP"]=[ele]
                    else :
                        item["FP"].append(ele)
        for ele in labels:
            if ele not in predicts:
                #漏了
                FN += 1
                if item.get("ori"):
                    if item.get("FN") is None:
                        item["FN"]=[ele]
                    else :
                        item["FN"].append(ele)
        
        if TP-TP_pre==len(predicts) and TP-TP_pre==len(labels):
            accnum+=1  # 整句话的预测都对了
        TP_pre=TP
    acc=accnum/len(predict_golden)
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1, acc


def tag2das(word_seq, tag_seq):
    assert len(word_seq)==len(tag_seq)
    das = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            slot = tag[2:].split('-')[-1]
            intent=""
            domain=""
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    # tag_seq[j][2:].split('+')[-1]==slot or tag_seq[j][2:] == tag[2:]
                    if word_seq[j].startswith('##'):
                        value += word_seq[j][2:]
                    else:
                        value += word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            if value.startswith('##'):
                value=value.replace('##','')
            das.append([intent, domain, slot, value])
        i += 1
    return das


def intent2das(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, domain, slot, value = re.split('\+', intent)
        triples.append([intent, domain, slot, value])
    return triples


def recover_intent(dataloader, intent_logits, tag_seq_id, tag_mask_tensor, ori_word_seq, new2ori):
    # tag_logits = [sequence_length, tag_dim]
    # intent_logits = [intent_dim]
    # tag_mask_tensor = [sequence_length]


    # print(tag_seq_id)
    # exit()
    das = []
    j=torch.argmax(intent_logits).item()

    intent, domain, slot, value = re.split('\+', dataloader.id2intent[j])
    das.append([intent, domain, slot, value])
    tags = []
    for j in range(len(tag_seq_id)):
            tags.append(dataloader.id2tag[tag_seq_id[j]])            
    tag_intent = tag2das(ori_word_seq, tags)
    das += tag_intent
    return das
