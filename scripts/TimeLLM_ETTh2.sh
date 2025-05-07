model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

# master_port=00098
master_port=25647
num_process=1
# batch_size=24
batch_size=4
d_model=32
d_ff=32
llm_model='pythia-14m'
llm_dim=128 # 对应于pythia-14m
softprompt_seq_len=128 # 来自p-tuning v2的数值：128
if_few_shot=1
comment='TimeLLM-ETTh2'
# note='woHardPrompt&MOMENT'
# note='woHardPrompt&MOMENT&InitializationOnlyDomainKnowledge&fewshot'
note='woHardPrompt&MOMENT&InitializationRandom&fewshot5%'
# note='woHardPrompt&Initialization'
# note='woHardPrompt&InitializationOnlyDomainKnowledge'
# note='self-consistency'
# note='SoftPrompt'
# note='woHardPrompt&Timesnet&newConcat'


/data/home/zhuomin/.conda/envs/softprompt/bin/accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --softprompt_seq_len $softprompt_seq_len \
  --llm_dim $llm_dim \
  --note $note \
  --if_few_shot $if_few_shot

# /data/home/zhuomin/.conda/envs/softprompt/bin/accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_192 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --lradj 'TST'\
#   --learning_rate 0.002 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment \
#   --llm_model $llm_model \
#   --softprompt_seq_len $softprompt_seq_len \
#   --llm_dim $llm_dim \
#   --note $note \
#   --if_few_shot $if_few_shot

# /data/home/zhuomin/.conda/envs/softprompt/bin/accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --lradj 'TST'\
#   --learning_rate 0.005 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment \
#   --llm_model $llm_model \
#   --softprompt_seq_len $softprompt_seq_len \
#   --llm_dim $llm_dim \
#   --note $note \
#   --if_few_shot $if_few_shot

/data/home/zhuomin/.conda/envs/softprompt/bin/accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.005 \
  --lradj 'TST'\
  --llm_layers $llama_layers \
  --train_epochs 20 \
  --patience 10 \
  --model_comment $comment \
  --llm_model $llm_model \
  --softprompt_seq_len $softprompt_seq_len \
  --llm_dim $llm_dim \
  --note $note \
  --if_few_shot $if_few_shot