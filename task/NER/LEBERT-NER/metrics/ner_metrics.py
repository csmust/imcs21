import torch
from collections import Counter
from utils.get_entity import get_entities

class SeqEntityScore(object):
    def __init__(self, id2label,markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'slot_acc': precision, 'slot_recall': recall, 'slot_f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):  # label_path[1, 1, 1, 6, 11, 1, 1, 1, 6, 1, 1, 1, 6, 1, ...]
            label_entities = get_entities(label_path, self.id2label,self.markup) #[['Symptom', 3, 4], ['Symptom', 17, 19], ['Symptom', 21, 22]]
            pre_entities = get_entities(pre_path, self.id2label,self.markup) # pre_path[6, 1, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
            return label_entities,pre_entities


