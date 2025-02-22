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
from models.jointbert import *

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
    def __init__(self, config,intent_num_labels,use_context):
        super(LEBertCrfForNer, self).__init__(config)
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim)
        self.bert = LEBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_loss_fct = torch.nn.CrossEntropyLoss()
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.intent_num_labels = intent_num_labels
        self.CNN_NLP= CNN_NLP(embed_dim=self.bert.config.hidden_size,num_filters=[self.bert.config.hidden_size]*3)
        self.context=use_context
        self.lstm_hidden_size = LSTM_HIDDEN_SIZE
        # self.lstm_hidden_size = self.bert.config.hidden_size
        self.use_hidden=True
        if USELSTM:
            self.enc_i = Intent_Enc(in_size= self.bert.config.hidden_size , lstm_hidden_size = self.lstm_hidden_size ) #  768  512
            self.dec_i = Intent_Dec(self.lstm_hidden_size)
            self.enc_s = Slot_Enc(self.bert.config.hidden_size ,self.lstm_hidden_size)
            self.dec_s = Slot_Dec(self.lstm_hidden_size)
        if self.context:
            # self.intent_hidden = nn.Linear(2 * (self.bert.config.hidden_size ) , self.hidden_units)
            # self.slot_hidden = nn.Linear(2 * (self.bert.config.hidden_size ),self.hidden_units)
            self.intent_hidden = nn.Linear(2 * (self.bert.config.hidden_size + self.lstm_hidden_size) , config.hidden_size)
            self.slot_hidden = nn.Linear(2 * (self.bert.config.hidden_size + self.lstm_hidden_size),config.hidden_size)
            self.intent_classifier = nn.Linear(config.hidden_size,self.intent_num_labels)
            self.slot_classifier = nn.Linear(config.hidden_size, config.num_labels)

        else:
            self.intent_hidden = nn.Linear((self.bert.config.hidden_size + self.lstm_hidden_size) , config.hidden_size)
            self.slot_hidden = nn.Linear((self.bert.config.hidden_size + self.lstm_hidden_size),config.hidden_size)
            self.intent_classifier = nn.Linear(config.hidden_size,self.intent_num_labels)
            self.slot_classifier = nn.Linear(config.hidden_size,config.num_labels)

        self.newselfattention1 = Self_Attention_Muti_Head(self.lstm_hidden_size*2, self.lstm_hidden_size*2, self.lstm_hidden_size*2, 1)
        self.newselfattention2 = Self_Attention_Muti_Head(self.lstm_hidden_size*2, self.lstm_hidden_size*2, self.lstm_hidden_size*2, 4)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.init_weights()
        nn.init.xavier_uniform_(self.intent_hidden.weight)
        nn.init.xavier_uniform_(self.slot_hidden.weight)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids, word_ids, word_mask, input_context_ids,context_attention_mask,token_type_ids_context,context_word_ids,
                context_word_mask,intent_tensor=None ,labels=None):
        word_embeddings = self.word_embeddings(word_ids) # 主要是这里
        context_embeddings = self.word_embeddings(context_word_ids)
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            word_embeddings=word_embeddings, word_mask=word_mask
        )

        bert_sequence_output = outputs[0]
        bert_pooled_output = outputs[1]
        bert_sequence_output=self.CNN_NLP(bert_sequence_output)
        hs = self.enc_s(bert_sequence_output)
        self.share_memory_s = hs.clone()
        hi = self.enc_i(bert_sequence_output)
        self.share_memory_i = hi.clone()
        self.posembedding1 = PositionalEncoding(self.lstm_hidden_size*2, 0.0, hi.size(1)).to(self.device)
        bimodel_slot_output = self.dec_s(hs ,  self.newselfattention1(self.posembedding1(self.share_memory_i.detach()),self.posembedding1(self.share_memory_s.detach())))
        # print(bimodel_slot_output.size())
        active_len_seq = attention_mask.sum(dim=-1)  # 尝试：含cls 和 seq TODO 另一种做法不含
        # print(active_len_seq)
        bimodel_intent_output = self.dec_i(hi , self.newselfattention2(self.posembedding1(self.share_memory_s.detach()),self.posembedding1(self.share_memory_i.detach())) ,active_len_seq )   # [batch , lstm_hidden_size]
        '''
        BERT 和 LSTM 结果进行拼接
        '''
        sequence_output = torch.cat((bert_sequence_output,bimodel_slot_output),dim=-1) # 8 51 1024
        pooled_output = torch.cat((bert_pooled_output,bimodel_intent_output),dim=-1) # 8 1024


        if self.context:
            context_outputs = self.bert(
                input_ids=input_context_ids, attention_mask=context_attention_mask, token_type_ids=token_type_ids_context,
                word_embeddings=context_embeddings, word_mask=context_word_mask
            )
            bert_context_seqout , bert_context_poolout = context_outputs[0] , context_outputs[1]
            bert_context_seqout=self.CNN_NLP(bert_context_seqout)
            chs = self.enc_s(bert_context_seqout)
            self.share_memory_cs = chs.clone()
            chi = self.enc_i(bert_context_seqout)
            self.share_memory_ci = chi.clone()
            # bcso = self.dec_s( chs ,self.share_memory_ci.detach())
            active_len_context = context_attention_mask.sum(dim=-1)
            self.posembedding2 = PositionalEncoding(LSTM_HIDDEN_SIZE*2, 0.0, chi.size(1)).to(self.device)
            bcio = self.dec_i(chi , self.newselfattention2(self.posembedding2(self.share_memory_cs.detach()),self.posembedding2(self.share_memory_ci.detach())) ,active_len_context)
            context_output = torch.cat((bert_context_poolout,bcio),dim=-1)
            sequence_output = torch.cat(
            # context_output.unsqueeze(1) torch.Size([8, 1, 768]) # .repeat(通道的重复倍数1, 行的重复倍数sequence_output.size(1), 列的重复倍数1) 后torch.Size([8, 51, 768])
            [
                context_output.unsqueeze(1).repeat(
                    1, sequence_output.size(1), 1),  # unsqueeze()函数起升维的作用
                sequence_output
            ],
            dim=-1)  # torch.Size([8, 51, 1536])
            pooled_output = torch.cat([context_output, pooled_output],
                                      dim=-1)

        if self.use_hidden:
            sequence_output = nn.functional.relu(
                self.slot_hidden(
                    self.dropout(sequence_output)))  # [8, 51, 1536]
            pooled_output = nn.functional.relu(
                self.intent_hidden(self.dropout(pooled_output)))  # [8,1536]


        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)


        outputs = (slot_logits,)
        if labels is not None:
            slot_loss = self.crf(emissions=slot_logits, tags=labels, mask=attention_mask)
            outputs = (-1 * slot_loss,) + outputs

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits,intent_tensor)
            outputs = outputs + (intent_loss,)
        outputs = outputs + (intent_logits,)
        return outputs  # (slot_loss), slot_logits , (intent_loss), intent_logits
        # return outputs  # (loss), scores


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
