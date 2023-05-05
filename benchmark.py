from model_components.dataloader import get_dataloader
from baseline import bert_cos_sim, double_bert, cross_seg, llama_cos_sim, two_level_trans
from transformers import BertModel, LlamaModel
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from model_components.validator import validate
from model_components.loss_func import FocalLoss, CrossEntroy

def train(dataset_name,
          model_name,
          sentence_bert_name,
          win_len,
          step_len,
          max_token_num,
          bbox_flag,
          sentence_bert_flag,
          device,
          batch_size,
          seg_model_name,
          weight_0,
          weight_1,
          alpha,
          gamma,
          loss_func_name,
          cos_sim_threhold,
          semantic_dim):
    best = [0, 0]
    dataloader_train = get_dataloader(dataset_name=dataset_name,
                                      model_name=model_name,
                                      sentence_bert_name=sentence_bert_name,
                                      win_len=win_len,
                                      step_len=step_len,
                                      max_token_num=max_token_num,
                                      bbox_flag=bbox_flag,
                                      sentence_bert_flag=sentence_bert_flag,
                                      device=device,
                                      batch_size=batch_size,
                                      goal='train')
    dataloader_dev = get_dataloader(dataset_name=dataset_name,
                                      model_name=model_name,
                                      sentence_bert_name=sentence_bert_name,
                                      win_len=win_len,
                                      step_len=step_len,
                                      max_token_num=max_token_num,
                                      bbox_flag=bbox_flag,
                                      sentence_bert_flag=sentence_bert_flag,
                                      device=device,
                                      batch_size=batch_size,
                                      goal='dev')
    dataloader_val = get_dataloader(dataset_name=dataset_name,
                                      model_name=model_name,
                                      sentence_bert_name=sentence_bert_name,
                                      win_len=win_len,
                                      step_len=step_len,
                                      max_token_num=max_token_num,
                                      bbox_flag=bbox_flag,
                                      sentence_bert_flag=sentence_bert_flag,
                                      device=device,
                                      batch_size=batch_size,
                                      goal='val')
    epoch = 10000
    weight_cross = torch.tensor([weight_0, weight_1]).to(device)
    loss_func_dict = {'cross': CrossEntroy(weight=weight_cross),
                     'focal': FocalLoss(alpha=alpha, gamma=gamma)}
    loss_func = loss_func_dict[loss_func_name]
    seg_model_dict = {'bert_cos_sim':1}
    if 'llama' not in model_name:
        backbone_model = BertModel.from_pretrained(model_name).to(device)
    else:
        backbone_model = LlamaModel.from_pretrained(model_name).to(device)

    seg_model = bert_cos_sim.BertCosSim(bert_model=backbone_model,
                                        threshold=cos_sim_threhold).to(device)
    for epoch_num in range(epoch):
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            seg_model(data)
            if seg_model_name in ['bert_cos_sim', 'llama_cos_sim', 'double_bert']:
                break
            pass

        validate()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='choi')
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--sentence_bert_name",
                        default='sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens')
    parser.add_argument("--win_len", default=2)
    parser.add_argument("--step_len", default=2)
    parser.add_argument("--max_token_num", default=512)
    parser.add_argument("--bbox_flag", default=False)
    parser.add_argument("--sentence_bert_flag", default=False)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--weight_0", default=1.0)
    parser.add_argument("--weight_1", default=1.0)
    parser.add_argument("--alpha", default=0.25)
    parser.add_argument("--gamma", default=2.0)
    parser.add_argument("--cos_sim_threhold", default=0)
    parser.add_argument("--loss_func_name", default='cross', choices=['cross', 'focal'])
    parser.add_argument("--seg_model_name", default='bert_cos_sim')
    parser.add_argument("--semantic_dim", default=768)
    args = parser.parse_args()
    print(args)
    semantic_dim = int(args.semantic_dim)
    cos_sim_threhold = float(args.cos_sim_threhold)
    dataset_name = args.dataset_name
    model_name = args.model_name
    sentence_bert_name = args.sentence_bert_name
    win_len = int(args.win_len)
    step_len = int(args.step_len)
    max_token_num = int(args.max_token_num)
    bbox_flag = True if args.bbox_flag == '1' else False
    sentence_bert_flag = True if args.sentence_bert_flag == '1' else False
    device = args.device
    batch_size = int(args.batch_size)
    weight_0 = float(args.weight_0)
    weight_1 = float(args.weight_1)
    alpha = float(args.alpha)
    gamma = float(args.gamma)
    seg_model_name = args.seg_model_name
    loss_func_name = args.loss_func_name
    train(dataset_name=dataset_name,
          model_name=model_name,
          sentence_bert_name=sentence_bert_name,
          win_len=win_len,
          step_len=step_len,
          max_token_num=max_token_num,
          bbox_flag=bbox_flag,
          sentence_bert_flag=sentence_bert_flag,
          device=device,
          batch_size=batch_size,
          seg_model_name=seg_model_name,
          weight_0=weight_0,
          weight_1=weight_1,
          alpha=alpha,
          gamma=gamma,
          loss_func_name=loss_func_name,
          cos_sim_threhold=cos_sim_threhold,
          semantic_dim=semantic_dim)









