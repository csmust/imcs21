import torch
from models.ner_model import BertSoftmaxForNer, LEBertSoftmaxForNer, LEBertCrfForNer, BertCrfForNer
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
import os
import numpy as np
from os.path import join
from loguru import logger
import time
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from processors.processor import LEBertProcessor, BertProcessor
# import json
from tqdm import tqdm
from metrics.ner_metrics import SeqEntityScore
import transformers
# from nervaluate import Evaluator
from seqeval.metrics import precision_score, recall_score, f1_score
# import sklearn
# from ..eval_ner import ner_eval
from sklearn import metrics
import json


def set_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument("--output_path", type=str, default='output/', help='模型与预处理数据的存放位置')
    parser.add_argument("--pretrain_embed_path", type=str, default='/Users/yangjianxin/Downloads/tencent-ailab-embedding-zh-d200-v0.2.0/tencent-ailab-embedding-zh-d200-v0.2.0.txt', help='预训练词向量路径')

    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'], help='损失函数类型')
    parser.add_argument('--add_layer', default=1, type=str, help='在bert的第几层后面融入词汇信息')
    parser.add_argument("--lr", type=float, default=1e-5, help='Bert的学习率')
    parser.add_argument("--crf_lr", default=1e-2, type=float, help="crf的学习率")
    parser.add_argument("--lstm_lr", default=1e-3, type=float, help="自定义的lstm的学习率")
    parser.add_argument("--myattention_lr", default=2e-2, type=float, help="自定义的注意力学习率")
    parser.add_argument("--adapter_lr", default=1e-3, type=float, help="crf的学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=32)
    parser.add_argument("--eval_step", type=int, default=2, help="训练多少步，查看验证集的指标")
    parser.add_argument("--max_seq_len", type=int, default=128, help="输入的最大长度")
    # parser.add_argument("--max_context_len", type=int, default=128*2+3, help="历史对话的最大长度")
    parser.add_argument("--max_word_num", type=int, default=3, help="每个汉字最多融合多少个词汇信息")
    parser.add_argument("--max_scan_num", type=int, default=10000, help="取预训练词向量的前max_scan_num个构造字典树")
    parser.add_argument("--data_path", type=str, default="datasets/resume/", help='数据集存放路径')
    # parser.add_argument("--train_file", type=str, default="datasets/cner/train.txt")
    # parser.add_argument("--dev_file", type=str, default="datasets/cner/dev.txt")
    # parser.add_argument("--test_file", type=str, default="datasets/cner/test.txt")
    parser.add_argument("--dataset_name", type=str, choices=['resume', "weibo", 'ontonote4', 'msra', 'dialoimc'], default='resume', help='数据集名称')
    parser.add_argument("--model_class", type=str, choices=['lebert-softmax', 'bert-softmax', 'bert-crf', 'lebert-crf'],
                        default='lebert-crf', help='模型类别')
    parser.add_argument("--pretrain_model_path", type=str, default="pretrain_model/bert-base-chinese")
    parser.add_argument("--overwrite", action='store_true', default=True, help="覆盖数据处理的结果")
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--load_word_embed", action='store_true', default=True, help='是否加载预训练的词向量')

    parser.add_argument('--markup', default='bio', type=str, choices=['bios', 'bio'], help='数据集的标注方式')
    parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.')
    parser.add_argument('--use_context', type=bool, default=True, help="是否利用对话历史")
    args = parser.parse_args()
    return args


def seed_everything(seed=42):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def get_optimizer(model, args, warmup_steps, t_total):
    # no_bert = ["word_embedding_adapter", "word_embeddings", "classifier",  "crf"]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     # bert no_decay
    #     {
    #         "params": [p for n, p in model.named_parameters()
    #                    if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0, 'lr': args.lr
    #     },
    #     # bert decay
    #     {
    #         "params": [p for n, p in model.named_parameters()
    #                    if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay, 'lr': args.lr
    #     },
    #     # other no_decay
    #     {
    #         "params": [p for n, p in model.named_parameters()
    #                    if any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0, "lr": args.adapter_lr
    #     },
    #     # other decay
    #     {
    #         "params": [p for n, p in model.named_parameters() if
    #                    any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay, "lr": args.adapter_lr
    #     }
    # ]
    # todo 检查
    embedding = ["word_embedding_adapter", "word_embeddings"]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    crf = ["crf"]
    lstm = ["lstm.weight"]
    attention = ["unflatselfattention.weight","_conv.weight","newselfattention1.k.weight","newselfattention1.q.weight","newselfattention1.v.weight","newselfattention2.k.weight","newselfattention2.q.weight","newselfattention2.v.weight"]
    no_bert=embedding+crf+lstm+attention
    optimizer_grouped_parameters = [
        # bert no_decay
        
        #第一个部分中的条件(not any(nd in n for nd in no_bert)检查参数的名称是否包含在no_bert列表中的任何一个元素中。如果参数名称不包含在no_bert列表中，则条件为真。
        #另外，如果参数名称为bert.embeddings.word_embeddings.weight，则条件也为真。
        #第二个部分中的条件any(nd in n for nd in no_decay)检查参数的名称是否包含在no_decay列表中的任何一个元素中。如果参数名称包含在no_decay列表中，则条件为真。
        {
            "params": [p for n, p in model.named_parameters()
                       if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and any(nd in n for nd in no_decay)],  # 
            "weight_decay": 0.0, 'lr': args.lr
        },
        # bert decay
        {
            "params": [p for n, p in model.named_parameters()
                       if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.lr
        },
        # crf lr
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in crf) and p.requires_grad],
         'weight_decay': args.weight_decay,"lr":args.crf_lr},
        # lstm lr
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in lstm) and p.requires_grad],
            'weight_decay': args.weight_decay,"lr":args.lstm_lr},
        # attention lr
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in attention) and p.requires_grad],
            'weight_decay': args.weight_decay,"lr":args.myattention_lr},

        # other no_decay
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in embedding) and n != 'bert.embeddings.word_embeddings.weight' and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": args.adapter_lr
        },
        # other decay
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in embedding) and n != 'bert.embeddings.word_embeddings.weight' and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr": args.adapter_lr
        }
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


def train(model, train_loader, dev_loader, test_loader, optimizer, scheduler, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    dev = 0
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            stime = time.time()
            step = epoch * len(train_loader) + batch_idx + 1
            input_ids = data['input_ids'].to(device)  #torch.Size([64, 128])
            token_type_ids = data['token_type_ids'].to(device) #torch.Size([64, 128])
            attention_mask = data['attention_mask'].to(device) #torch.Size([64, 128]) 1是真实的，0是padding的tensor([[1, 1, 1,  ..., 0, 0, 0],
            label_ids = data['label_ids'].to(device)  #tensor([[1, 1, 1,  ..., 1, 1, 1],  torch.Size([64, 128])
            intent_tensor = data['intent_ids'].flatten().to(device)
            input_context_ids = data['input_context_ids'].to(device)
            context_attention_mask = data['context_mask'].to(device)
            token_type_ids_context = data['token_type_ids_context'].to(device)
            context_word_ids = data['context_word_ids'].to(device)
            context_word_mask = data['context_word_mask'].to(device)
            

            # 不同模型输入不同
            if args.model_class == 'bert-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)  #torch.Size([64, 128, 3])
                word_mask = data['word_mask'].to(device) #torch.Size([64, 128, 3])
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index, label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device) 
                # loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
                slot_loss, slot_logits,intent_loss,intent_logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask,input_context_ids,context_attention_mask,token_type_ids_context,context_word_ids ,context_word_mask,intent_tensor,label_ids)
            
            # loss = (slot_loss+intent_loss).mean()  # 对多卡的loss取平均
            slot_loss=slot_loss.mean()
            intent_loss=intent_loss.mean()
            loss=slot_loss+intent_loss

            # 梯度累积
            loss = loss / args.grad_acc_step
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if step % args.grad_acc_step == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            # 评测验证集和测试集上的指标
            if step % args.eval_step == 0:
                
                logger.info('train intent loss: {},in step {} epoch {}'.format(intent_loss.item(), step, epoch + 1))
                logger.info('train slot loss: {},in step {} epoch {}'.format(slot_loss.item(), step, epoch + 1))

                # print("validating...")
                # logger.info('------evaluate 验证集------')
                # dev_result = evaluate(args, model, dev_loader)

                # # logger.info('dev slot_loss: {}'.format(dev_result['slot_loss']))
                # # logger.info('dev slot_f1: {}'.format(dev_result['slot_f1']))
                # # logger.info('dev slot_precision: {}'.format(dev_result['slot_acc']))
                # # logger.info('dev slot_recall: {}'.format(dev_result['slot_recall']))
                # writer.add_scalar('dev slot_loss', dev_result['slot_loss'], step)
                # writer.add_scalar('dev slot_f1', dev_result['slot_f1'], step)
                # writer.add_scalar('dev slot_precision', dev_result['slot_acc'], step)
                # writer.add_scalar('dev slot_recall', dev_result['slot_recall'], step)
                # writer.add_scalar('dev intent_loss', dev_result['intent_loss'], step)
                # writer.add_scalar('dev intent_f1', dev_result['intent_f1'], step)
                # writer.add_scalar('dev intent_precision', dev_result['intent_precision'], step)
                # writer.add_scalar('dev intent_recall', dev_result['intent_recall'], step)
                # writer.add_scalar('dev intent_acc', dev_result['intent_acc'], step)
                print("testing...")
                logger.info('------evaluate 测试集------')
                test_result = evaluate(args, model, test_loader, iftest=False)

                # logger.info('test slot_loss: {}'.format(test_result['slot_loss']))
                # logger.info('test slot_f1: {}'.format(test_result['slot_f1']))
                # logger.info('test slot_precision: {}'.format(test_result['slot_acc']))
                # logger.info('test slot_recall: {}'.format(test_result['slot_recall']))
                writer.add_scalar('test slot_loss', test_result['slot_loss'], step)
                writer.add_scalar('test slot_f1', test_result['slot_f1'], step)
                writer.add_scalar('test slot_precision', test_result['slot_acc'], step)
                writer.add_scalar('test slot_recall', test_result['slot_recall'], step)
                writer.add_scalar('test intent_loss', test_result['intent_loss'], step)
                writer.add_scalar('test intent_f1', test_result['intent_f1'], step)
                writer.add_scalar('test intent_precision', test_result['intent_precision'], step)
                writer.add_scalar('test intent_recall', test_result['intent_recall'], step)
                writer.add_scalar('test intent_acc', test_result['intent_acc'], step)


                
    #             if best < test_result['overall_f1']:
    #                 best = test_result['overall_f1']
    #                 dev = dev_result['overall_f1']
    #                 logger.info('higher f1 of testset is {}, dev is {} in step {} epoch {}'.format(best, dev, step, epoch+1))
    #                 # save_path = join(args.output_path, 'checkpoint-{}'.format(step))
    #                 model_to_save = model.module if hasattr(model, 'module') else model
    #                 model_to_save.save_pretrained(args.output_path)
    #             etime = time.time()
    #             logger.info('step {} time cost: {}\n\n'.format(step , etime - stime))
    #             model.train()
    # logger.info('best f1 of test is {}, dev is {}'.format(best, dev))

                if best < test_result['overall_f1']:
                    best = test_result['overall_f1']
                    logger.info('higher f1 of testset is {},in step {} epoch {}'.format(best, step, epoch+1))
                    # save_path = join(args.output_path, 'checkpoint-{}'.format(step))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_path)
                etime = time.time()
                logger.info('step {} time cost: {}\n\n'.format(step , etime - stime))
                model.train()
    logger.info('best f1 of test is {}'.format(best))


def evaluate(args, model, dataloader,iftest=False):
    model.eval()
    device = args.device
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    # Eval!
    logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = {}".format(len(dataloader)))
    # logger.info("  Batch size = {}".format(args.batch_size_eval))
    eval_loss = 0.0  #
    eval_slot_loss = 0.0
    eval_intent_loss = 0.0
    final_preds, final_labels = [], []    # 槽
    predict_all = np.array([], dtype=int)  # 意图
    labels_all = np.array([], dtype=int)  # 意图
    
    pred_slot_value,label_slot_value=[],[]  # 槽值 例如 [['Symptom-咳嗽','Symptom-感冒'],[第二个句子的],[第三个句子的…………]]
    pred_intent_str,label_intent_str=[],[] # 意图字面值  例如[[], [] , []……………………]
    predicts_overall=[]
    goldens_overall=[]
    text_overall=[]
    output=[]
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)

            intent_tensor = data['intent_ids'].flatten().to(device)
            input_context_ids = data['input_context_ids'].to(device)
            context_attention_mask = data['context_mask'].to(device)
            token_type_ids_context = data['token_type_ids_context'].to(device)
            context_word_ids = data['context_word_ids'].to(device)
            context_word_mask = data['context_word_mask'].to(device)
            text = data['text']#['医生：你好，咳嗽是连声咳吗？有痰吗？有没流鼻涕，鼻塞？', '医生：咳嗽有几天了？']
            text = [x[3:] for x in text] #['你好，咳嗽是连声咳吗？有痰吗？有没流鼻涕，鼻塞？', '咳嗽有几天了？']
            text_overall.extend(text)
            # 不同模型输入不同
            if args.model_class == 'bert-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index,
                                     label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                # loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
                #return (slot_loss), slot_logits , (intent_loss), intent_logits
                slot_loss, slot_logits,intent_loss,intent_logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask,input_context_ids,context_attention_mask,token_type_ids_context,context_word_ids ,context_word_mask,intent_tensor,label_ids)

            slot_loss=slot_loss.mean()
            intent_loss=intent_loss.mean()

            loss=slot_loss+intent_loss
            eval_loss += loss
            eval_slot_loss += slot_loss
            eval_intent_loss += intent_loss

            input_lens = (torch.sum(input_ids != 0, dim=-1) - 5).tolist()   # 减去padding的[CLS]与[SEP]
            if args.model_class in ['lebert-crf', 'bert-crf']:
                preds = model.crf.decode(slot_logits, attention_mask).squeeze(0)
                preds = preds[:, 4:].tolist()  # 减去padding的[CLS]  预测时候没算 CLS 医 生 ：
            else:
                preds = torch.argmax(logits, dim=2)[:, 4:].tolist()  # 减去padding的[CLS]
            pred_intent = torch.argmax(intent_logits, dim=1).tolist()
            label_ids = label_ids[:, 4:].tolist()   # 减去padding的[CLS]    标签也 没算 CLS 医 生 ：
            intent_ids = intent_tensor.tolist()
            # preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            # label_ids = label_ids.cpu().numpy().tolist()
            for i in range(len(label_ids)):
                input_len = input_lens[i]
                pred = preds[i][:input_len]
                label = label_ids[i][:input_len]
                label_entities,pre_entities=metric.update(pred_paths=[pred], label_paths=[label])  #return label_entities,pre_entities
                final_preds.append([args.id2label[_] for _ in pred])
                final_labels.append([args.id2label[_] for _ in label])
                pred_slot_value.append([mf[0]+"-"+text[i][int(mf[1]):int(mf[2])+1] for mf in pre_entities])
                label_slot_value.append([mf[0]+"-"+text[i][int(mf[1]):int(mf[2])+1] for mf in label_entities])

            #意图评分
            predic=torch.max(intent_logits.data, 1)[1].cpu().numpy() #array([10,  0])
            labels=intent_tensor.data.cpu().numpy()  #array([13, 13])
            labels_all = np.append(labels_all, labels) #将labels添加到labels_all中
            predict_all = np.append(predict_all, predic) #将predic添加到predict_all中
            assert len(labels) == len(predic)
            for j in range(len(labels)):
                pred_intent_str.append([args.id2intent[predic[j]]])  #[['Inform-Drug_Recommendation'], ['Inform-Etiology']]
                label_intent_str.append([args.id2intent[labels[j]]])  #[['Inform-Drug_Recommendation'], ['Inform-Etiology']]





    logger.info("\n")
    eval_slot_loss = eval_slot_loss / len(dataloader)
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['slot_loss'] = eval_slot_loss

    logger.info("***** slot Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)

    logger.info("***** slot Entity results(单独计算实体槽名) *****")
    for key in sorted(entity_info.keys()):
        logger.info("******* %s slot results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)


    #意图评分
    logger.info("***** intent Eval results *****")
    p = metrics.precision_score(labels_all, predict_all, average='macro') #计算精确率
    r = metrics.recall_score(labels_all, predict_all, average='macro') #计算召回率
    f1 = metrics.f1_score(labels_all, predict_all, average='macro') #计算f1
    acc = metrics.accuracy_score(labels_all, predict_all) #计算准确率
    if iftest:
        report = metrics.classification_report(labels_all, predict_all, target_names=args.id2intent, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # results['intent_report'] = report
        # results['intent_confusion'] = confusion
        logger.info("***** test intent report *****")
        logger.info(report)
        logger.info("***** test intent confusion *****")
        logger.info(confusion)
    results['intent_acc'] = acc
    results['intent_f1'] = f1
    results['intent_precision'] = p
    results['intent_recall'] = r
    results['intent_loss'] = eval_intent_loss / len(dataloader)
    logger.info("intent acc: {:.4f} - intent f1: {:.4f} -  intent precision: {:.4f} - intent recall: {:.4f} - intent loss: {:.4f}".\
                format(acc, f1, p, r,results['intent_loss']))

    # evaluator = Evaluator(final_labels, final_preds,
    #                       tags=['Symptom', 'Medical_Examination', 'Drug', 'Drug_Category', 'Operation'],
    #                       loader="list")
    # e_results, e_results_by_tag = evaluator.evaluate()
    # print(e_results)
    # print(e_results_by_tag)

    try:
        print('seqeval p/r/f1: {}/{}/{}'.format(
            precision_score(final_labels, final_preds),
            recall_score(final_labels, final_preds),
            f1_score(final_labels, final_preds),
        ))
    except Exception as e:
        print(e)

    #overall评分
    logger.info("***** overall Eval results *****")
    predicts_overall=[x+y for x,y in zip(pred_slot_value,pred_intent_str)]
    goldens_overall=[x+y for x,y in zip(label_slot_value,label_intent_str)]
    for i in range(len(predicts_overall)):
        output.append({"text":text_overall[i],"predict":predicts_overall[i],"golden":goldens_overall[i]})
    

    overall_p, overall_r, overall_f1, overall_acc = calculateF1(output,iftest)
    logger.info("overall acc: {:.4f} - overall f1: {:.4f} -  overall precision: {:.4f} - overall recall: {:.4f}".\
                format(overall_acc, overall_f1, overall_p, overall_r))

    results['overall_precision'] = overall_p
    results['overall_recall'] = overall_r
    results['overall_f1'] = overall_f1
    results['overall_acc'] = overall_acc
    if iftest:
        #将output列表保存到json
        with open('output.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


    

    return results

def calculateF1(output,iftest=False):
    TP, FP, FN = 0, 0, 0
    accnum=0
    TP_pre=0
    for item in output:
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
                if iftest:
                    if item.get("FP") is None:
                        item["FP"]=[ele]
                    else :
                        item["FP"].append(ele)
        for ele in labels:
            if ele not in predicts:
                #漏了
                FN += 1
                if iftest:
                    if item.get("FN") is None:
                        item["FN"]=[ele]
                    else :
                        item["FN"].append(ele)
        
        if TP-TP_pre==len(predicts) and TP-TP_pre==len(labels):
            accnum+=1  # 整句话的预测都对了
        TP_pre=TP
    acc=accnum/len(output)
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1, acc


MODEL_CLASS = {
    'lebert-softmax': LEBertSoftmaxForNer,
    'lebert-crf': LEBertCrfForNer,
    'bert-softmax': BertSoftmaxForNer,
    'bert-crf': BertCrfForNer
}
PROCESSOR_CLASS = {
    'lebert-softmax': LEBertProcessor,
    'lebert-crf': LEBertProcessor,
    'bert-softmax': BertProcessor,
    'bert-crf': BertProcessor
}


def main(args):
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)  
    # 数据处理器
    processor = PROCESSOR_CLASS[args.model_class](args, tokenizer)
    args.id2label = processor.label_vocab.idx2token #['[PAD]', 'O', 'B-Drug', 'B-Drug_Category', 'B-Medical_Examination', 'B-Operation', 'B-S
    args.ignore_index = processor.label_vocab.convert_token_to_id('[PAD]') #0
    #intent
    args.id2intent = processor.intent_vocab.idx2token
    # 初始化模型配置
    config = BertConfig.from_pretrained(args.pretrain_model_path)
    config.num_labels = processor.label_vocab.size
    intent_num_labels = processor.intent_vocab.size
    config.loss_type = args.loss_type           #ce
    if args.model_class in ['lebert-softmax', 'lebert-crf']:
        config.add_layer = args.add_layer # 在1层添加
        config.word_vocab_size = processor.word_embedding.shape[0] # 词汇表大小
        config.word_embed_dim = processor.word_embedding.shape[1] # 词向量维度
    # 初始化模型
    model = MODEL_CLASS[args.model_class].from_pretrained(args.pretrain_model_path,config=config ,intent_num_labels=intent_num_labels ,use_context=args.use_context).to(args.device)  
    # 初始化模型的词向量
    if args.model_class in ['lebert-softmax', 'lebert-crf'] and args.load_word_embed:
        logger.info('initialize word_embeddings with pretrained embedding')
        model.word_embeddings.weight.data.copy_(torch.from_numpy(processor.word_embedding))

    # 训练
    if args.do_train:
        # 加载数据集
        train_dataset = processor.get_train_data()#<processors.dataset.NERDataset object
        # train_dataset = train_dataset[:8]
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_dataset = processor.get_dev_data()
        # dev_dataset = dev_dataset[:4]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        test_dataset = processor.get_test_data()
        # test_dataset = test_dataset[:4]
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        t_total = len(train_dataloader) // args.grad_acc_step * args.epochs
        warmup_steps = int(t_total * args.warmup_proportion)
        # optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        # )
        optimizer, scheduler = get_optimizer(model, args, warmup_steps, t_total)
        train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args)

    # 测试集上的指标 TODO
    if args.do_eval:
        # 加载验证集
        dev_dataset = processor.get_dev_data()
        # dev_dataset = dev_dataset[:4]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        # 加载测试集
        test_dataset = processor.get_test_data()
        # test_dataset = test_dataset[:4]
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        model = MODEL_CLASS[args.model_class].from_pretrained(args.output_path, intent_num_labels=intent_num_labels,use_context=args.use_context).to(args.device)
        model.eval()
        logger.info("验证保存的模型，将输出结果保存到output.json")
        result = evaluate(args, model, dev_dataloader)
        logger.info('devset slot_precision:{}, slot_recall:{}, slot_f1:{}, slot_loss:{}'.format(result['slot_acc'], result['slot_recall'], result['slot_f1'], result['slot_loss'].item()))
        logger.info('devset intent_precision:{}, intent_recall:{},intent_acc:{}, intent_f1:{}, intent_loss:{}'.format(result['intent_precision'], result['intent_recall'],result["intent_acc"], result['intent_f1'], result['intent_loss'].item()))
        # 测试集上的指标
        result = evaluate(args, model, test_dataloader,iftest=True)
        logger.info('testset slot_precision:{}, slot_recall:{}, slot_f1:{}, slot_loss:{}'.format(result['slot_acc'], result['slot_recall'], result['slot_f1'], result['slot_loss'].item()))
        logger.info('testset intent_precision:{}, intent_recall:{},intent_acc:{}, intent_f1:{}, intent_loss:{}'.format(result['intent_precision'], result['intent_recall'],result["intent_acc"], result['intent_f1'], result['intent_loss'].item()))


if __name__ == '__main__':
    # 设置参数
    args = set_train_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

    pretrain_model = 'mengzi' if 'mengzi' in args.pretrain_model_path else 'bert-base'
    args.output_path = join(args.output_path, args.dataset_name, args.model_class, pretrain_model, 'load_word_embed' if args.load_word_embed else 'not_load_word_embed')
    print(args.output_path)
    args.train_file = join(args.data_path, 'train.json')
    args.dev_file = join(args.data_path, 'dev.json')
    args.test_file = join(args.data_path, 'test.json')
    args.label_path = join(args.data_path, 'labels.txt')
    args.intent_path = join(args.data_path, 'intents.txt')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        logger.info(args)
        writer = SummaryWriter(args.output_path)

    main(args)
