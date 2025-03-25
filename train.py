import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader,args,qid2type):
    dataset=args.dataset
    num_epochs=args.epochs
    mode=args.mode
    run_eval=args.eval_each_epoch
    output=args.output

    optim = torch.optim.Adamax(model.parameters())

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    if mode=='q_debias':
        topq=args.topq
        keep_qtype=args.keep_qtype
    elif mode=='v_debias':
        topv=args.topv
        top_hint=args.top_hint
    elif mode=='q_v_debias':
        topv=args.topv
        top_hint=args.top_hint
        topq=args.topq
        keep_qtype=args.keep_qtype
        qvp=args.qvp

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (o, v, q, a, b, v2, q2, a2, b2) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):

            total_step += 1
            #########################################
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            v2 = Variable(v2).cuda()
            q2 = Variable(q2).cuda()
            a2 = Variable(a2).cuda()
            b2 = Variable(b2).cuda()
            #########################################
            if mode=='updn':

                pred, loss, _, _ = model(v, q, a, b, None, v2, q2, a2, b2)
                # print(loss)
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

        if mode=='updn':
            total_loss /= len(train_loader.dataset)
        else:
            total_loss /= len(train_loader.dataset) * 2
        train_score = 100 * train_score / len(train_loader.dataset)

        # if epoch >= args.epoch and epoch <= args.endepoch:
        #     model.train(False)
        #     resize_bias_all(model=model, dataloader=train_loader, epoch=epoch+1, stop_many=args.stop_many, w=args.w)
        #     model.train(True)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score

def reset_bias_all(train_dset, processed_dict, every_question_type, min_value, max_value, epoch, stop_many, w):
    from collections import defaultdict, Counter
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    answer_voc_size = train_dset.num_ans_candidates

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # ? reset bias score two
    for i, ex in enumerate(train_dset.entries):
        # if stop_many < epoch and ex['type'] == 'number':
        #     continue
        if ex['answer']['scores'] is not None:
            global_bias = 1 - (processed_dict[i] - min_value) / (max_value - min_value)
            local_bias = 1 - (processed_dict[i] - every_question_type[ex['answer']['question_type']]['min']) / (every_question_type[ex['answer']['question_type']]['max'] - every_question_type[ex['answer']['question_type']]['min'])
            # w = 1
            temp = w * global_bias + (1 - w) * local_bias
            # ex['answer']['new_scores'] = ex['answer']['scores'] # * new_add
            ex['answer']['scores'] = torch.mul(ex['answer']['scores'], temp)
            # ex['answer']['scores'] = torch.mul(ex['answer']['scores'], global_bias)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for i, ex in enumerate(train_dset.entries):
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ex in train_dset.entries:
        q_type = ex["answer"]["question_type"]
        ex["bias"] = question_type_to_prob_array[q_type]

    # for i, ex in enumerate(train_dset.entries):                 # * new_add
    #     if ex['answer']['scores'] is not None:                  # * new_add
    #         ex['answer']['scores'] = ex['answer']['new_scores'] # * new_add

# ? 重新调整训练集的bias，不仅仅是全部类型，而且要把每一个quesition_type结合过来，需要设置两个超级参数，分别为w_1,w_2
# * 不应该修改答案的内容,而是修改bias的值
def resize_bias_all(model, dataloader, epoch, stop_many=2, w=1.0):
    recording_loss = dict()
    for _, (o, v, q, a, b, _, _, _, _) in tqdm(enumerate(dataloader), ncols=100,
                                                   desc="bias", total=len(dataloader)):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        b = Variable(b, requires_grad=False).cuda()
        _, _, loss_record, _ = model(v, q, a, b, None)
        
        recording_loss.update(zip(o.tolist(), loss_record.tolist()))
    # ? 计算每个分类的最大值、均值、最小值
    every_question_type = dict()
    for question_type in list(set([ex['answer']['question_type'] for ex in dataloader.dataset.entries])):
        question_type_list = [i for i, ex in enumerate(dataloader.dataset.entries) if ex['answer']['question_type']==question_type]
        every_recording_loss_list = [recording_loss[i] for i in question_type_list]
        every_question_type[question_type] = {
            'min': min(every_recording_loss_list),
            'max': max(every_recording_loss_list),
        }
    values = recording_loss.values()
    min_value = min(values)
    max_value = max(values)
    reset_bias_all(dataloader.dataset, recording_loss, every_question_type, min_value, max_value, epoch, stop_many, w)

def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, b, qids, _, _ in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred, _, _, _ = model(v, q, None, None, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results
