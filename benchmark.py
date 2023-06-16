from model_components.dataloader import get_dataloader
from baseline import bert_cos_sim, double_bert, cross_seg,\
    llama_cos_sim, two_level_trans, sentence_bert, sector
from model_config.fig_seg import FigSeg
from model_config.encoder_seg import EncoderSeg
from transformers import BertModel, LlamaModel
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from model_components.validator import validate
from model_components.loss_func import FocalLoss, CrossEntroy, LossFigSeg

torch.manual_seed(3407)

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
          cos_sim_threshold,
          semantic_dim,
          dev_step,
          feature_type,
          token_encoder_flag,
          sentence_encoder_flag,
          partial_encoder_flag,
          llama_flag):
    best = [100, 0, 0, 0, 0]
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
    weight_cross = torch.tensor([weight_0, weight_1]).to('cuda')
    loss_func_dict = {'cross': CrossEntroy(device=device,
                                           dataset_name=dataset_name,
                                           dataloader=dataloader_train,
                                           model_type=seg_model_name),
                      'focal': FocalLoss(alpha=alpha, gamma=gamma),
                      'fig_seg': LossFigSeg(device=device,
                                            dataset_name=dataset_name,
                                            dataloader=dataloader_train,
                                            model_type=seg_model_name,
                                            loss_func_name='cross')}
    if seg_model_name != 'fig_seg':
        loss_func = loss_func_dict[loss_func_name]
    else:
        loss_func = loss_func_dict['fig_seg']

    loss_all = []
    if 'llama' not in model_name:
        backbone_model = BertModel.from_pretrained(model_name).to(device)
    else:
        backbone_model = LlamaModel.from_pretrained(model_name).to(device)

    seg_model_dict = {'bert_cos_sim': bert_cos_sim.BertCosSim(bert_model=backbone_model,
                                                              threshold=cos_sim_threshold),
                      'double_bert': double_bert.DoubleBert(bert_model=backbone_model,
                                                            threshold=cos_sim_threshold),
                      'llama_cos_sim': llama_cos_sim.LlamaCosSim(bert_model=backbone_model,
                                                                 threshold=cos_sim_threshold,
                                                                 feature_type=feature_type),
                      'sentence_bert': sentence_bert.SentenceBertCosSim(cos_sim_threshold),
                      'two_level': two_level_trans.TwoLevelTrans(sim_dim=semantic_dim),
                      'cross_seg': cross_seg.CrossSeg(bert_model=backbone_model,
                                                      sim_dim=semantic_dim),
                      'sector': sector.Sector(sim_dim=semantic_dim),
                      'fig_seg': FigSeg(token_encoder_flag=token_encoder_flag,
                                        sentence_encoder_flag=sentence_encoder_flag,
                                        partial_encoder_flag=partial_encoder_flag,
                                        sim_dim=semantic_dim,
                                        bert_model=backbone_model,
                                        llama_flag=llama_flag,
                                        bbox_flag=bbox_flag),
                      'encoder_seg': EncoderSeg(token_encoder_flag=token_encoder_flag,
                                                sentence_encoder_flag=sentence_encoder_flag,
                                                partial_encoder_flag=partial_encoder_flag,
                                                sim_dim=semantic_dim,
                                                bert_model=backbone_model,
                                                llama_flag=llama_flag,
                                                bbox_flag=bbox_flag)
                      }
    seg_model = seg_model_dict[seg_model_name].to(device)
    seg_model.train()
    try:
        frozen_layer = seg_model.bert_model
        for param in frozen_layer.parameters():
            param.requires_grad = False
    except:
        pass
    try:
        optimizer = torch.optim.AdamW(seg_model.parameters(), lr=4e-5)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.5,
                                      patience=2,
                                      verbose=True)
    except:
        pass

    if seg_model_name in ['bert_cos_sim', 'llama_cos_sim',
                          'double_bert', 'sentence_bert']:
        epoch = 1
    else:
        epoch = 100

    for epoch_num in range(epoch):
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            # if step >= 3:
            #     break
            if seg_model_name in ['bert_cos_sim', 'llama_cos_sim',
                                  'double_bert', 'sentence_bert']:
                break
            else:
                output = seg_model(data)
                if seg_model_name != 'fig_seg':
                    prob = output['prob']
                    loss_value = loss_func(prob, data['label_seg'].view(-1))
                else:
                    loss_value = loss_func(output, data)

                loss_all.append(loss_value.item())
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()


            if (step+1) % dev_step == 0:
                dev_output = validate(seg_model=seg_model,
                                      dataloader=dataloader_dev,
                                      loss_func=loss_func,
                                      seg_model_name=seg_model_name)

        val_output = validate(seg_model=seg_model,
                              dataloader=dataloader_val,
                              loss_func=loss_func,
                              seg_model_name=seg_model_name)
        try:
            scheduler.step(val_output['loss'])
        except:
            pass
        pk = val_output['pk']
        windiff = val_output['windiff']
        p = val_output['p']
        r = val_output['r']
        f = val_output['f']
        print(epoch_num)
        try:
            print('train_loss: ', sum(loss_all) / len(loss_all))
        except:
            pass
        print('val_loss: ', val_output['loss'])
        print('pk: ', pk)
        print('windiff: ', windiff)
        print('p: ', p)
        print('r: ', r)
        print('f: ', f)
        if pk <= best[0]:
            best[0] = pk
            best[1] = windiff
            best[2] = p
            best[3] = r
            best[4] = f
        print('best')
        print('pk: ', best[0])
        print('windiff: ', best[1])
        print('p: ', best[2])
        print('r: ', best[3])
        print('f: ', best[4])
        save_folder_path = 'log/' + seg_model_name + '/' + dataset_name
        save_folder_flag = os.path.exists(save_folder_path)
        if not save_folder_flag:
            os.makedirs(save_folder_path)

        save_file_path = save_folder_path + '/' + 'result_' + feature_type + '.txt'
        with open(save_file_path, 'a+') as file:
            file.write('====================\n')
            file.write(str(epoch_num) + '\n'
                       + 'pk: ' + str(pk) + '\n'
                       + 'windiff: ' + str(windiff) + '\n'
                       + 'p: ' + str(p) + '\n'
                       + 'r: ' + str(r) + '\n'
                       + 'f: ' + str(f) + '\n'
                       + 'best_pk: ' + str(best[0]) + '\n'
                       + 'best_windiff: ' + str(best[1]) + '\n'
                       + 'best_p: ' + str(best[2]) + '\n'
                       + 'best_r: ' + str(best[3]) + '\n'
                       + 'best_f: ' + str(best[4]) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='50', choices=['choi', '50',
                                                                 'fr',  'fi',
                                                                 'city', 'diseases'])
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--sentence_bert_name",
                        default='sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens')
    parser.add_argument("--win_len", default=8)
    parser.add_argument("--step_len", default=7)
    parser.add_argument("--max_token_num", default=300)
    parser.add_argument("--bbox_flag", default='0')
    parser.add_argument("--sentence_bert_flag", default='1')
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--weight_0", default=1.0)
    parser.add_argument("--weight_1", default=1.0)
    parser.add_argument("--alpha", default=0.75)
    parser.add_argument("--gamma", default=2.0)
    parser.add_argument("--dev_step", default=10000)
    parser.add_argument("--cos_sim_threshold", default=0.5)
    parser.add_argument("--loss_func_name", default='cross', choices=['cross',
                                                                      'focal'])
    parser.add_argument("--seg_model_name", default='encoder_seg',
                        choices=['bert_cos_sim', 'double_bert', 'llama_cos_sim',
                                 'sentence_bert', 'two_level', 'cross_seg', 'fig_seg',
                                 'encoder_seg', 'sector'])
    '''double_bert 求bert和sentenceBert最大值SECTOR
       two_level transformer平方 win:2 step:1
       对于我们的模型 win:8 step:7
    '''
    parser.add_argument("--semantic_dim", default=768)
    parser.add_argument("--feature_type", default='mean', choices=['max', 'mean'])
    parser.add_argument("--token_encoder_flag", default='1')
    parser.add_argument("--sentence_encoder_flag", default='1')
    parser.add_argument("--partial_encoder_flag", default='1')
    args = parser.parse_args()
    print(args)
    llama_flag = True if 'llama' in args.model_name else False
    token_encoder_flag = True if args.token_encoder_flag == '1' else False
    sentence_encoder_flag = True if args.sentence_encoder_flag == '1' else False
    partial_encoder_flag = True if args.partial_encoder_flag == '1' else False
    feature_type = args.feature_type
    semantic_dim = int(args.semantic_dim)
    dev_step = int(args.dev_step)
    cos_sim_threshold = float(args.cos_sim_threshold)
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
          cos_sim_threshold=cos_sim_threshold,
          semantic_dim=semantic_dim,
          dev_step=dev_step,
          feature_type=feature_type,
          token_encoder_flag=token_encoder_flag,
          sentence_encoder_flag=sentence_encoder_flag,
          partial_encoder_flag=partial_encoder_flag,
          llama_flag=llama_flag)









