export CUDA_VISIBLE_DEVICES=0
model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llm_layers=32

master_port=00097  # 00097
num_process=1
batch_size=4
d_model=64
d_ff=64

llm_model='pythia-14m'
llm_dim=128
n_heads=4

# llm_model='pythia-1b'
# llm_dim=2048
# n_heads=8
# llm_dim=256 # 和d_model一样不行调用LLM模型参数的时候会报错

softprompt_seq_len=128

comment='TimeLLM-Exchange'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data Exchange \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --softprompt_seq_len $softprompt_seq_len \
  --llm_dim $llm_dim \
  --n_heads $n_heads

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data Exchange \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --softprompt_seq_len $softprompt_seq_len \
  --llm_dim $llm_dim \
  --n_heads $n_heads

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data Exchange \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --softprompt_seq_len $softprompt_seq_len \
  --llm_dim $llm_dim \
  --n_heads $n_heads

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data Exchange \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --softprompt_seq_len $softprompt_seq_len \
  --llm_dim $llm_dim \
  --n_heads $n_heads