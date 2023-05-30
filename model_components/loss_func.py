import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        one_hot = torch.zeros(targets.size(0), 2).to(targets.device)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        targets = one_hot
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha
            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CrossEntroy(nn.Module):
    def __init__(self, device, dataset_name, model_type, dataloader):
        super(CrossEntroy, self).__init__()
        self.model_type = model_type
        self.dataloader = dataloader
        self.weight = self.get_weight(dataset_name, device)
        self.loss_function = nn.CrossEntropyLoss(weight=self.weight)
        # self.loss_function = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.loss_function(inputs, targets)

    def get_weight(self, dataset_name, device):
        dataset_name_dict = {'choi': 'data/train_choi.csv',
                             '50': 'data/train_wiki50.csv',
                             'fr': 'data/train_fr.csv',
                             'fi': 'data/train_fi.csv',
                             'city': 'data/train_wikicity.csv',
                             'diseases': 'data/train_wikidiseases.csv',
                             }
        label_all = []
        if self.model_type != 'fig_seg':
            csv = pd.read_csv(dataset_name_dict[dataset_name])
            for label in csv['label_seg']:
                label_all += literal_eval(label)
        else:
            print('compute weight of loss func')
            for data in tqdm(self.dataloader):
                for epoch_index in range(data['label_fig'].shape[0]):
                    label_all += data['label_fig'][epoch_index].tolist()

        num_of_0 = label_all.count(0)
        num_of_1 = label_all.count(1)
        weight_0 = (num_of_0 + num_of_1) / (2.0 * num_of_0)
        weight_1 = (num_of_0 + num_of_1) / (2.0 * num_of_1)
        weight = torch.tensor([weight_0, weight_1]).to(device)

        weight = compute_class_weight(class_weight='balanced',
                                      classes=[0, 1],
                                      y=label_all)
        weight = torch.from_numpy(weight).float().to(device)
        # weight = torch.tensor([1.0, 10000.0]).float().to(device)
        return weight

class LossFigSeg(torch.nn.Module):
    def __init__(self, device,
                 dataset_name,
                 model_type,
                 dataloader,
                 loss_func_name):
        super(LossFigSeg, self).__init__()
        self.loss_func_name = loss_func_name
        self.loss_cos_sim = torch.nn.CosineEmbeddingLoss()
        self.loss_circle = CircleLoss()
        if loss_func_name == 'cross':
            self.loss_classification = CrossEntroy(device=device,
                                           dataset_name=dataset_name,
                                           dataloader=dataloader,
                                           model_type=model_type)
        else:
            self.loss_classification = FocalLoss(gamma=2, alpha=0.25)

    def forward(self, output_model, data):
        if self.loss_func_name == 'cross':
            loss = self.forward_cross(output_model, data)
        else:
            loss = self.forward_focal(output_model, data)

        loss_cos_sim_token = self.forward_cos_sim(output_model, data, 'token_feature')
        loss_cos_sim_sentence = self.forward_cos_sim(output_model, data, 'sentence_feature')
        loss_fig_circle = self.loss_circle(output_model['logit_fig'].view(-1, 2),
                                           data['label_cos_sim_matrix'].view(-1) ^ 1).mean()
        loss_seg_circle = self.loss_circle(output_model['logit_seg'].view(-1, 2),
                                           data['label_seg'].view(-1)).mean()

        return loss_seg_circle + \
               + loss

    def forward_cross(self, output_model, data):
        loss_fig = self.loss_classification(output_model['prob_fig'].view(-1, 2),
                                            data['label_cos_sim_matrix'].view(-1) ^ 1)
        loss_prob = self.loss_classification(output_model['logit_seg'].view(-1, 2),
                                             data['label_seg'].view(-1))
        return loss_prob

    def forward_focal(self, output_model, data):
        loss_fig = self.loss_classification(output_model['prob_fig'].view(-1, 2),
                                            data['label_cos_sim_matrix'].view(-1) ^ 1)
        loss_prob = self.loss_classification(output_model['seg_prob'].view(-1, 2),
                                             data['label_seg'].view(-1))
        return loss_prob + loss_fig

    def forward_cos_sim(self, output_model, data, label_str):
        input_1 = []
        input_2 = []
        batch_num = output_model[label_str].shape[0]
        sentence_num = output_model[label_str].shape[1]
        for batch_index in range(batch_num):
            for sentence_index in range(sentence_num - 1):
                input_1.append(output_model[label_str][batch_index][sentence_index])
                input_2.append(output_model[label_str][batch_index][sentence_index+1])

        label = data['label_seg'].clone().view(-1)
        label[label == 1] = -1
        label[label == 0] = 1
        loss = self.loss_cos_sim(torch.stack(input_1), torch.stack(input_2), label)
        return loss

    def post_process(self, sim_fig):
        batch_size = sim_fig.shape[0]
        result = []
        for batch_index in range(batch_size):
            result.append([])

class CircleLoss(torch.nn.Module):
    def __init__(self):
        super(CircleLoss, self).__init__()

    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
             本文。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e30
        y_pred_pos = y_pred - (1 - y_true) * 1e30
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        return neg_loss + pos_loss

    def forward(self, inputs, targets):
        one_hot = torch.zeros(targets.size(0), 2).to(targets.device)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        targets = one_hot
        loss = self.multilabel_categorical_crossentropy(targets, inputs)
        return loss



