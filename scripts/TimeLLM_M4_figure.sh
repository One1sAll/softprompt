model_name=TimeLLM

# train_epochs=50
train_epochs=40
llama_layers=32
batch_size=4
learning_rate=0.001

master_port=00097
num_process=1

llm_model='pythia-14m'
llm_dim=128
n_heads=4

softprompt_seq_len=128

comment='TimeLLM-M4'

# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_m4.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Monthly' \
#   --model_id m4_Monthly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --llm_layers $llama_layers \
#   --d_model 32 \
#   --d_ff 32 \
#   --patch_len 1 \
#   --stride 1 \
#   --batch_size $batch_size \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --loss 'SMAPE' \
#   --train_epochs 13 \
#   --model_comment $comment \
#   --llm_model $llm_model \
#   --softprompt_seq_len $softprompt_seq_len \
#   --llm_dim $llm_dim \
#   --n_heads $n_heads

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port figure.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model 16 \
  --d_ff 32 \
  --patch_len 1 \
  --stride 1 \
  --batch_size $batch_size \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs 7 \
  --model_comment $comment \
  --llm_model $llm_model \
  --softprompt_seq_len $softprompt_seq_len \
  --llm_dim $llm_dim \
  --n_heads $n_heads

# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_m4.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Weekly' \
#   --model_id m4_Weekly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --llm_layers $llama_layers \
#   --d_model 32 \
#   --d_ff 32 \
#   --patch_len 1 \
#   --stride 1 \
#   --batch_size $batch_size \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --loss 'SMAPE' \
#   --train_epochs 8 \
#   --model_comment $comment \
#   --llm_model $llm_model \
#   --softprompt_seq_len $softprompt_seq_len \
#   --llm_dim $llm_dim \
#   --n_heads $n_heads

# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_m4.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id m4_Daily \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --llm_layers $llama_layers \
#   --d_model 16 \
#   --d_ff 16 \
#   --patch_len 1 \
#   --stride 1 \
#   --batch_size $batch_size \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --loss 'SMAPE' \
#   --train_epochs 9 \
#   --model_comment $comment \
#   --llm_model $llm_model \
#   --softprompt_seq_len $softprompt_seq_len \
#   --llm_dim $llm_dim \
#   --n_heads $n_heads

# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_m4.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Quarterly' \
#   --model_id m4_Quarterly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --llm_layers $llama_layers \
#   --d_model 64 \
#   --d_ff 64 \
#   --patch_len 1 \
#   --stride 1 \
#   --batch_size $batch_size \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --loss 'SMAPE' \
#   --train_epochs $train_epochs \
#   --model_comment $comment \
#   --llm_model $llm_model \
#   --softprompt_seq_len $softprompt_seq_len \
#   --llm_dim $llm_dim \
#   --n_heads $n_heads


# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_m4.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Hourly' \
#   --model_id m4_Hourly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --llm_layers $llama_layers \
#   --d_model 32 \
#   --d_ff 32 \
#   --patch_len 1 \
#   --stride 1 \
#   --batch_size $batch_size \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --loss 'SMAPE' \
#   --train_epochs 9 \
#   --model_comment $comment \
#   --llm_model $llm_model \
#   --softprompt_seq_len $softprompt_seq_len \
#   --llm_dim $llm_dim \
#   --n_heads $n_heads