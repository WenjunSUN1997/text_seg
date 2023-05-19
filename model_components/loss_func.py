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
        # self.weight = self.get_weight(dataset_name, device)
        # self.loss_function = nn.CrossEntropyLoss(weight=self.weight)
        self.loss_function = nn.CrossEntropyLoss()

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

        # num_of_0 = label_all.count(0)
        # num_of_1 = label_all.count(1)
        # weight_0 = (num_of_0 + num_of_1) / (2.0 * num_of_0)
        # weight_1 = (num_of_0 + num_of_1) / (2.0 * num_of_1)
        # weight = torch.tensor([weight_0, weight_1]).to(device)

        weight = compute_class_weight(class_weight='balanced',
                                      classes=[0, 1],
                                      y=label_all)
        weight = torch.tensor(weight).to(device)
        return weight

class LossFigSeg(torch.nn.Module):
    def __init__(self, device,
                 dataset_name,
                 model_type,
                 dataloader,
                 loss_func_name):
        super(LossFigSeg, self).__init__()
        self.loss_func_name = loss_func_name
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

        return loss

    def forward_cross(self, output_model, data):
        loss_fig = self.loss_classification(output_model['prob_fig'].view(-1, 2),
                                            data['label_cos_sim_matrix'].view(-1) ^ 1)
        loss_prob = self.loss_classification(output_model['seg_prob'].view(-1, 2),
                                             data['label_seg'].view(-1))
        return loss_prob + loss_fig

    def forward_focal(self, output_model, data):
        loss_fig = self.loss_classification(output_model['prob_fig'].view(-1, 2),
                                            data['label_cos_sim_matrix'].view(-1) ^ 1)
        loss_prob = self.loss_classification(output_model['seg_prob'].view(-1, 2),
                                             data['label_seg'].view(-1))
        return loss_prob + loss_fig


    def post_process(self, sim_fig):
        batch_size = sim_fig.shape[0]
        result = []
        for batch_index in range(batch_size):
            result.append([])



