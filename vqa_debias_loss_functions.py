from collections import OrderedDict, defaultdict, Counter

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import inspect


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """computes log(sigmoid(logits)), log(1-sigmoid(logits))"""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))


def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm


class DebiasLossFn(nn.Module):
    """General API for our loss functions"""

    def forward(self, hidden, logits, bias, labels):
        """
        :param hidden: [batch, n_hidden] hidden features from the last layer in the model
        :param logits: [batch, n_answers_options] sigmoid logits for each answer option
        :param bias: [batch, n_answers_options]
          bias probabilities for each answer option between 0 and 1
        :param labels: [batch, n_answers_options]
          scores for each answer option, between 0 and 1
        :return: Scalar loss
        """
        raise NotImplementedError()

    def to_json(self):
        """Get a json representation of this loss function.

        We construct this by looking up the __init__ args
        """
        cls = self.__class__
        init = cls.__init__
        if init is object.__init__:
            return []  # No init args

        init_signature = inspect.getargspec(init)
        if init_signature.varargs is not None:
            raise NotImplementedError("varags not supported")
        if init_signature.keywords is not None:
            raise NotImplementedError("keywords not supported")
        args = [x for x in init_signature.args if x != "self"]
        out = OrderedDict()
        out["name"] = cls.__name__
        for key in args:
            out[key] = getattr(self, key)
        return out

class Plain(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss *= labels.size(1)
        return loss

# class Focal(DebiasLossFn):
#     def forward(self, hidden, logits, bias, labels):
#         # import pdb;pdb.set_trace()
#         focal_logits=torch.log(F.softmax(logits,dim=1)+1e-5) * ((1-F.softmax(bias,dim=1))*(1-F.softmax(bias,dim=1)))
#         loss=F.binary_cross_entropy_with_logits(focal_logits,labels)
#         loss*=labels.size(1)
#         return loss

# class ReweightByInvBias(DebiasLossFn):
#     def forward(self, hidden, logits, bias, labels):
#         # Manually compute the binary cross entropy since the old version of torch always aggregates
#         log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
#         loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob)
#         weights = (1 - bias)
#         loss *= weights  # Apply the weights
#         return loss.sum() / weights.sum()

# class BiasProduct(DebiasLossFn):
#     def __init__(self, smooth=True, smooth_init=-1, constant_smooth=0.0):
#         """
#         :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
#         :param smooth_init: How to initialize `a`
#         :param constant_smooth: Constant to add to the bias to smooth it
#         """
#         super(BiasProduct, self).__init__()
#         self.constant_smooth = constant_smooth
#         self.smooth_init = smooth_init
#         self.smooth = smooth
#         if smooth:
#             self.smooth_param = torch.nn.Parameter(
#               torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
#         else:
#             self.smooth_param = None

#     def forward(self, hidden, logits, bias, labels):
#         smooth = self.constant_smooth
#         if self.smooth:
#             smooth += F.sigmoid(self.smooth_param)

#         # Convert the bias into log-space, with a factor for both the
#         # binary outputs for each answer option
#         bias_lp = torch.log(bias + smooth)
#         bias_l_inv = torch.log1p(-bias + smooth)

#         # Convert the the logits into log-space with the same format
#         log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
#         # import pdb;pdb.set_trace()

#         # Add the bias
#         log_prob += bias_lp
#         log_one_minus_prob += bias_l_inv

#         # Re-normalize the factors in logspace
#         log_prob, log_one_minus_prob = renormalize_binary_logits(log_prob, log_one_minus_prob)

#         # Compute the binary cross entropy
#         loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0)
#         return loss

class Focal(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        # 计算 focal logits
        softmax_logits = F.softmax(logits, dim=1)
        softmax_bias = F.softmax(bias, dim=1)
        focal_weight = (1 - softmax_bias) ** 2  # focal loss weight
        focal_logits = torch.log(softmax_logits + 1e-5) * focal_weight
        
        # 计算每个元素的二元交叉熵损失，不进行汇总
        loss = F.binary_cross_entropy_with_logits(focal_logits, labels, reduction='none')
        
        # 将每个样本的损失从多个类别汇总为一个标量
        # 这里我们对每个样本的损失取平均（可以改为 sum 如果你想要总和）
        loss_per_sample = loss.mean(dim=1)  # [batch_size]
        
        return loss_per_sample

class ReweightByInvBias(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        # Manually compute the binary cross entropy since the old version of torch always aggregates
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        
        # 计算每个元素的二元交叉熵损失
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob)
        
        # 计算权重 (1 - bias)，并确保权重不会为负
        weights = (1 - bias).clamp(min=0)  # 防止权重为负
        
        # 应用权重到损失
        weighted_loss = loss * weights
        
        # 将每个样本的损失从多个类别汇总为一个标量
        # 这里我们对每个样本的损失取平均（可以改为 sum 如果你想要总和）
        loss_per_sample = weighted_loss.sum(dim=1) / (weights.sum(dim=1) + 1e-8)  # [batch_size]
        
        return loss_per_sample

class BiasProduct(DebiasLossFn):
    def __init__(self, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(BiasProduct, self).__init__()
        self.constant_smooth = constant_smooth
        self.smooth_init = smooth_init
        self.smooth = smooth
        if smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    def forward(self, hidden, logits, bias, labels):
        smooth = self.constant_smooth
        if self.smooth:
            smooth += F.sigmoid(self.smooth_param)

        # Convert the bias into log-space, with a factor for both the
        # binary outputs for each answer option
        bias_lp = torch.log(bias + smooth)
        bias_l_inv = torch.log1p(-bias + smooth)

        # Convert the the logits into log-space with the same format
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        # import pdb;pdb.set_trace()

        # Add the bias
        log_prob += bias_lp
        log_one_minus_prob += bias_l_inv

        # Re-normalize the factors in logspace
        log_prob, log_one_minus_prob = renormalize_binary_logits(log_prob, log_one_minus_prob)

        # Compute the binary cross entropy
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1)
        return loss

'''
这段代码定义了一个继承了DebiasLossFn的LearnedMixin类。该类的构造函数接受四个参数，分别是权重(w)、平滑性(smooth)、初始平滑值(smooth_init)和常量平滑值(constant_smooth)。其中，加权规则可以通过调整w的大小来控制。

forward方法实现了前向传播操作。具体来说，它接受4个输入参数：hidden表示提取的句子特征、logits表示预测的答案分数、bias表示问题类型偏差、labels表示真实的标签。首先从hidden中提取一个二分类器的置信度factor，以及将偏差转换到log空间(bias = torch.log(bias))，并对logits与bias进行加权修正得到最终的log概率(logits = bias + log_probs)。根据预测得分和真实标签计算损失loss。在训练过程中还可以添加熵惩罚项entropy，用于缓解模型的过拟合。

此外，如果设置了smooth为True，就会使用一个可学习参数的sigmoid函数，来进一步平滑bias的值。注意，该学习参数在代码中被命名为self.smooth_param，并且在构造函数中已经初始化为了某一个值。整个类的作用主要是对模型进行偏差消减，进而提高其泛化能力和鲁棒性。
'''
# class LearnedMixin(DebiasLossFn):
#     def __init__(self, w, smooth=True, smooth_init=-1, constant_smooth=0.0):
#         """
#         :param w: Weight of the entropy penalty
#         :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
#         :param smooth_init: How to initialize `a`
#         :param constant_smooth: Constant to add to the bias to smooth it
#         """
#         super(LearnedMixin, self).__init__()
#         self.w = w
#         # self.w=0
#         self.smooth_init = smooth_init
#         self.constant_smooth = constant_smooth
#         self.bias_lin = torch.nn.Linear(1024, 1)
#         self.smooth = smooth
#         if self.smooth:
#             self.smooth_param = torch.nn.Parameter(
#               torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
#         else:
#             self.smooth_param = None

#     def forward(self, hidden, logits, bias, labels):
#         factor = self.bias_lin.forward(hidden)  # [batch, 1]
#         factor = F.softplus(factor)

#         bias = torch.stack([bias, 1 - bias], 2)  # [batch, n_answers, 2]

#         # Smooth
#         bias += self.constant_smooth
#         if self.smooth:
#             soften_factor = F.sigmoid(self.smooth_param)
#             bias = bias + soften_factor.unsqueeze(1)

#         bias = torch.log(bias)  # Convert to logspace

#         # Scale by the factor
#         # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
#         bias = bias * factor.unsqueeze(1)

#         log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
#         log_probs = torch.stack([log_prob, log_one_minus_prob], 2)

#         # Add the bias in
#         logits = bias + log_probs

#         # Renormalize to get log probabilities
#         log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

#         # Compute loss
#         loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0)

#         # Re-normalized version of the bias
#         bias_norm = elementwise_logsumexp(bias[:, :, 0], bias[:, :, 1])
#         bias_logprob = bias - bias_norm.unsqueeze(2)

#         # Compute and add the entropy penalty
#         entropy = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean()
#         return loss + self.w * entropy


class LearnedMixin(DebiasLossFn):
    def __init__(self, w, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param w: Weight of the entropy penalty
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.w = w
        # self.w=0
        self.smooth_init = smooth_init
        self.constant_smooth = constant_smooth
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.smooth = smooth
        if self.smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    # self.debias_loss_fn(joint_repr, logits, bias, labels)
    def forward(self, hidden, logits, bias, labels):
        # * 计算偏置值
        factor = self.bias_lin.forward(hidden)  # [batch, 1]
        factor = F.softplus(factor)

        # * 一个关注偏见，一个关注偏见之外的内容
        bias = torch.stack([bias, 1 - bias], 2)  # [batch, n_answers, 2]

        # * 进行平滑操作
        # Smooth
        bias += self.constant_smooth
        if self.smooth:
            soften_factor = F.sigmoid(self.smooth_param)
            bias = bias + soften_factor.unsqueeze(1)
        
        # bias = torch.clamp(bias, min=1e-7, max=1-1e-7) # * 避免出现负数和0
        bias = torch.clamp(bias, min=1e-9, max=1-1e-9) # ? best
        # bias = torch.clamp(bias, min=1e-13, max=1-1e-13)

        # * 获得对数概率
        bias = torch.log(bias)  # Convert to logspace 

        # Scale by the factor
        # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
        bias = bias * factor.unsqueeze(1)

        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)

        log_probs = torch.stack([log_prob, log_one_minus_prob], 2)

        # Add the bias in
        logits = bias + log_probs

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        # Compute loss
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1)

        # Re-normalized version of the bias
        bias_norm = elementwise_logsumexp(bias[:, :, 0], bias[:, :, 1])
        bias_logprob = bias - bias_norm.unsqueeze(2)

        # Compute and add the entropy penalty
        entropy = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean(1)
        loss = loss + self.w * entropy
        return loss
