import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np

def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def process(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        att = nn.functional.softmax(att, 1)

        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        return joint_repr

    def forward(self, v, q, labels, bias, v_mask=None, v2=None, q2=None, labels2=None, bias2=None):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q) # torch.Size([512, 14, 300])
        q_emb = self.q_emb(w_emb)  # [batch, q_dim] # torch.Size([512, 1024])

        att = self.v_att(v, q_emb) # torch.Size([512, 36, 1])
        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att= mask_softmax(att,v_mask) # torch.Size([512, 2048])

        v_emb = (att * v).sum(1)  # [batch, v_dim] torch.Size([512, 2048])

        q_repr = self.q_net(q_emb) # torch.Size([512, 1024])
        v_repr = self.v_net(v_emb) # torch.Size([512, 1024])
        joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)

        loss2 = None
        loss3 = None
        if v2 is not None:
            joint_repr2 = self.process(v2, q2)
            logits2 = self.classifier(joint_repr2)
            loss2 = self.debias_loss_fn(joint_repr2, logits2, bias2, labels).mean(0)
            loss3 = self.criterion(joint_repr, joint_repr2, batch_size=joint_repr.size(0))
        if labels is not None:
            loss_record = self.debias_loss_fn(joint_repr, logits, bias, labels)
            loss1 = loss_record.mean(0)
            if loss2 is None:
                loss = loss1
            else:
                loss = loss2 + loss3
        else:
            loss1 = None
            loss = None
            loss_record = None
        return logits, loss, loss_record, w_emb
        # return logits, loss1, att
    
        if labels is not None:
            # loss = self.debias_loss_fn(joint_repr, logits, bias, labels) + criterion(q_emb, q_repr, batch_size=q_emb.size(0))
            loss_record = self.debias_loss_fn(joint_repr, logits, bias, labels)
            loss = loss_record.mean(0)
            # loss = loss_record
        else:
            loss_record = None
            loss = None
        return logits, loss, loss_record, w_emb
    
    def criterion(self,out_1,out_2,tau_plus=0.1,batch_size=512,beta=1.0,estimator='easy', temperature=0.5):
        out_1 = nn.functional.normalize(out_1, dim=-1)
        out_2 = nn.functional.normalize(out_2, dim=-1)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if estimator=='hard':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()
        return loss

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
