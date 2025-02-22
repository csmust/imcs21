import torch
import torch.nn as nn
import torch.nn.functional as F
from models.crf import CRF
from models.lebert import LEBertModel
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from losses.focal_loss import FocalLoss
from transformers import BertModel, BertPreTrainedModel, BertConfig


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, ignore_index, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=ignore_index)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=ignore_index)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrfForNer(BertPreTrainedModel, ):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs =self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


class LEBertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(LEBertSoftmaxForNer, self).__init__(config)
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim)
        self.num_labels = config.num_labels
        self.bert = LEBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, word_ids, word_mask, ignore_index, labels=None): #ignore_index 自带padding功能，  labels中的ignore_index（pad）不会参与loss计算
        word_embeddings = self.word_embeddings(word_ids)
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            word_embeddings=word_embeddings, word_mask=word_mask
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=ignore_index)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=ignore_index)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class LEBertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(LEBertCrfForNer, self).__init__(config)
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim)
        self.bert = LEBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, word_ids, word_mask, labels=None):
        word_embeddings = self.word_embeddings(word_ids)
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            word_embeddings=word_embeddings, word_mask=word_mask
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores


if __name__ == '__main__':
    # pretrain_model_path = '../pretrain_model/test'
    pretrain_model_path = '../pretrain_model/bert-base-chinese'
    input_ids = torch.randint(0,100,(4,10))
    token_type_ids = torch.randint(0,1,(4,10))
    attention_mask = torch.randint(1,2,(4,10))
    word_embeddings = torch.randn((4,10,5,200))
    word_mask = torch.randint(0,1,(4,10,5))
    config = BertConfig.from_pretrained(pretrain_model_path, num_labels=20)
    config.word_embed_dim = 200
    config.num_labels = 20
    config.loss_type = 'ce'
    config.add_layer = 0
    model = LEBertSoftmaxForNer.from_pretrained(pretrain_model_path, config=config)
    # model = LEBertSoftmaxForNer(config=config)
    labels = torch.randint(0,3,(4,10))
    # model = BertModel(n_layers=2, d_model=64, d_ff=256, n_heads=4,
    #              max_seq_len=128, vocab_size=1000, pad_id=0, dropout=0.1)
    loss, logits = model(
        input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
        word_embeddings=word_embeddings, word_mask=word_mask, labels=labels
    )
    print(loss)
