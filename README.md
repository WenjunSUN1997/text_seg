# to test the code
python -u benchmark.py

--dataset_name fr (fr, fi, diseases)

--model_name decapoda-research/llama-7b-hf 

--bbox_flag 0 

--seg_model_name encoder_seg 

--device cuda:0 (you can change) 

--win_len 8 

--step_len 7

--batch_size 4(you can change)

if you meet some error from transformer library, try to renew it to version 4.30.2
and install another library:sentence-transformers

pip install sentence-transformers

