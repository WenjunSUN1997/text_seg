from benchmark import train

if __name__ == "__main__":
    config_list = {'bert_cos_sim':{'model_name': 'bert-base-uncased',
                                   'sentence_bert_name':'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens',
                                   'win_len': 2,
                                   'step_len': 1,
                                   'max_token_num': 512,
                                   'bbox_flag': False,
                                   'sentence_bert_flag': True,
                                   'device': 'cuda:2',
                                   'batch_size': 4,
                                   'weight_0': 1.0,
                                   'weight_1': 1.0,
                                   'alpha': 0.25,
                                   'gamma': 2.0,
                                   'dev_step': 1000,
                                   'cos_sim_threshold': 0.5,
                                   'loss_func_name': 'cross',
                                   'semantic_dim': 768,
                                   'feature_type': 'max',
                                   'seg_model_name': 'bert_cos_sim'},

                   'double_bert': {'model_name': 'bert-base-uncased',
                                   'sentence_bert_name':'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens',
                                   'win_len': 2,
                                   'step_len': 1,
                                   'max_token_num': 512,
                                   'bbox_flag': False,
                                   'sentence_bert_flag': True,
                                   'device': 'cuda:2',
                                   'batch_size': 4,
                                   'weight_0': 1.0,
                                   'weight_1': 1.0,
                                   'alpha': 0.25,
                                   'gamma': 2.0,
                                   'dev_step': 1000,
                                   'cos_sim_threshold': 0.5,
                                   'loss_func_name': 'cross',
                                   'semantic_dim': 768,
                                   'feature_type': 'max',
                                   'seg_model_name': 'double_bert'},
                   'sentence_bert': {'model_name': 'bert-base-uncased',
                                   'sentence_bert_name': 'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens',
                                   'win_len': 2,
                                   'step_len': 1,
                                   'max_token_num': 512,
                                   'bbox_flag': False,
                                   'sentence_bert_flag': True,
                                   'device': 'cuda:2',
                                   'batch_size': 4,
                                   'weight_0': 1.0,
                                   'weight_1': 1.0,
                                   'alpha': 0.25,
                                   'gamma': 2.0,
                                   'dev_step': 1000,
                                   'cos_sim_threshold': 0.5,
                                   'loss_func_name': 'cross',
                                   'semantic_dim': 768,
                                   'feature_type': 'max',
                                   'seg_model_name': 'sentence_bert'},
                   'llama_cos_sim': {'model_name': 'decapoda-research/llama-7b-hf',
                                     'sentence_bert_name': 'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens',
                                     'win_len': 2,
                                     'step_len': 1,
                                     'max_token_num': 512,
                                     'bbox_flag': False,
                                     'sentence_bert_flag': True,
                                     'device': 'cuda:2',
                                     'batch_size': 4,
                                     'weight_0': 1.0,
                                     'weight_1': 1.0,
                                     'alpha': 0.25,
                                     'gamma': 2.0,
                                     'dev_step': 1000,
                                     'cos_sim_threshold': 0,
                                     'loss_func_name': 'cross',
                                     'semantic_dim': 768,
                                     'feature_type': 'max',
                                     'seg_model_name': 'llama_cos_sim'}}

    dataset_choices = ['choi', '50', 'fr', 'fi', 'city', 'diseases']
    for dataset_name in dataset_choices:
        seg_model_name_list = list(config_list.keys())
        for seg_model_name in seg_model_name_list:
            print(dataset_name)
            print(seg_model_name)
            model_name = config_list[seg_model_name]['model_name']
            sentence_bert_name = config_list[seg_model_name]['sentence_bert_name']
            win_len = config_list[seg_model_name]['win_len']
            step_len = config_list[seg_model_name]['step_len']
            max_token_num = config_list[seg_model_name]['max_token_num']
            bbox_flag = False
            sentence_bert_flag = config_list[seg_model_name]['sentence_bert_flag']
            device = 'cuda:2'
            batch_size = 4
            weight_0 = config_list[seg_model_name]['weight_0']
            weight_1 = config_list[seg_model_name]['weight_1']
            alpha = config_list[seg_model_name]['alpha']
            gamma = config_list[seg_model_name]['gamma']
            loss_func_name = config_list[seg_model_name]['loss_func_name']
            cos_sim_threshold = config_list[seg_model_name]['cos_sim_threshold']
            semantic_dim = config_list[seg_model_name]['semantic_dim']
            dev_step = config_list[seg_model_name]['dev_step']
            feature_type = config_list[seg_model_name]['feature_type']
            if seg_model_name in ['bert_cos_sim', 'double_bert',
                                  'sentence_bert', 'two_level',
                                  'cross_seg'] and dataset_name == 'fr':
                model_name = 'camembert-base'

            if seg_model_name in ['bert_cos_sim', 'double_bert',
                                  'sentence_bert', 'two_level',
                                  'cross_seg'] and dataset_name == 'fi':
                model_name = 'TurkuNLP/bert-base-finnish-cased-v1'

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
                  token_encoder_flag=False,
                  sentence_encoder_flag=False,
                  partial_encoder_flag=False,
                  llama_flag=False
                  )
