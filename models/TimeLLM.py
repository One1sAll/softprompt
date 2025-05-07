from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from models.prefix_encoder import PrefixEncoder
from models.TimesNet import  TimesNet
import os
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
import sys
from momentfm import MOMENTPipeline
print("Module search path:", sys.path)
transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2) # 从倒数第2个维度开始扁平化，将最后两个维度合并
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x shape :   B, N, llm_dim, input_ids_num+..
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# class Model(nn.Module):

#     def __init__(self, configs, patch_len=16, stride=8):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.pred_len = configs.pred_len
#         self.seq_len = configs.seq_len
#         self.d_ff = configs.d_ff
#         self.top_k = 5
#         self.d_llm = configs.llm_dim
#         self.patch_len = configs.patch_len
#         self.stride = configs.stride
#         print('Time-LLM.py class model init()')

#         if configs.llm_model == 'LLAMA':
#             # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
#             self.llama_config = LlamaConfig.from_pretrained('/mnt/chenzhm39/Time-LLM/models/llama-7b')
#             # self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
#             self.llama_config.num_hidden_layers = configs.llm_layers
#             self.llama_config.output_attentions = True
#             self.llama_config.output_hidden_states = True
#             print('llama_config')
#             try:
#                 self.llm_model = LlamaModel.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
#                     '/mnt/chenzhm39/Time-LLM/models/llama-7b',
#                     # 'huggyllama/llama-7b',
#                     trust_remote_code=True, # 信任远程代码
#                     local_files_only=True, # 只从本地文件系统加载模型，而不是从 Hugging Face 的模型库中下载。这意味着模型文件应该已经存在于本地文件系统中。
#                     config=self.llama_config,
#                     # load_in_4bit=True
#                 )
#             except EnvironmentError:  # downloads model from HF is not already done
#                 print("Local model files not found. Attempting to download...")
#                 print('loading llamamodel failed')
#                 self.llm_model = LlamaModel.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
#                     'huggyllama/llama-7b',
#                     trust_remote_code=True,
#                     local_files_only=False, # 文件不在本地，需要从hugging face下载
#                     config=self.llama_config,
#                     # load_in_4bit=True
#                 )
#             try:
#                 self.tokenizer = LlamaTokenizer.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
#                     '/mnt/chenzhm39/Time-LLM/models/llama-7b/tokenizer.model',
#                     # 'huggyllama/llama-7b',
#                     trust_remote_code=True,
#                     local_files_only=True
#                 )
#             except EnvironmentError:  # downloads the tokenizer from HF if not already done
#                 print("Local tokenizer files not found. Atempting to download them..")
#                 print('loading llamatokenizer failed')
#                 self.tokenizer = LlamaTokenizer.from_pretrained(
#                     # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
#                     'huggyllama/llama-7b',
#                     trust_remote_code=True,
#                     local_files_only=False
#                 )
#         elif configs.llm_model == 'GPT2':
#             self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

#             self.gpt2_config.num_hidden_layers = configs.llm_layers
#             self.gpt2_config.output_attentions = True
#             self.gpt2_config.output_hidden_states = True
#             try:
#                 self.llm_model = GPT2Model.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=True,
#                     config=self.gpt2_config,
#                 )
#             except EnvironmentError:  # downloads model from HF is not already done
#                 print("Local model files not found. Attempting to download...")
#                 self.llm_model = GPT2Model.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=False,
#                     config=self.gpt2_config,
#                 )

#             try:
#                 self.tokenizer = GPT2Tokenizer.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=True
#                 )
#             except EnvironmentError:  # downloads the tokenizer from HF if not already done
#                 print("Local tokenizer files not found. Atempting to download them..")
#                 self.tokenizer = GPT2Tokenizer.from_pretrained(
#                     'openai-community/gpt2',
#                     trust_remote_code=True,
#                     local_files_only=False
#                 )
#         elif configs.llm_model == 'BERT':
#             # self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
#             self.bert_config = BertConfig.from_pretrained('/root/autodl-tmp/softprompt/models/bert-large-uncased')

#             self.bert_config.num_hidden_layers = configs.llm_layers
#             self.bert_config.output_attentions = True
#             self.bert_config.output_hidden_states = True
#             try:
#                 self.llm_model = BertModel.from_pretrained(
#                     # 'google-bert/bert-base-uncased',
#                     '/root/autodl-tmp/softprompt/models/bert-large-uncased',
#                     trust_remote_code=True,
#                     local_files_only=True,
#                     config=self.bert_config,
#                 )
#             except EnvironmentError:  # downloads model from HF is not already done
#                 print("Local model files not found. Attempting to download...")
#                 self.llm_model = BertModel.from_pretrained(
#                     # 'google-bert/bert-base-uncased',
#                     '/root/autodl-tmp/softprompt/models/bert-large-uncased',
#                     trust_remote_code=True,
#                     local_files_only=False,
#                     config=self.bert_config,
#                 )

#             try:
#                 self.tokenizer = BertTokenizer.from_pretrained(
#                     # 'google-bert/bert-base-uncased',
#                     '/root/autodl-tmp/softprompt/models/bert-large-uncased',
#                     trust_remote_code=True,
#                     local_files_only=True
#                 )
#             except EnvironmentError:  # downloads the tokenizer from HF if not already done
#                 print("Local tokenizer files not found. Atempting to download them..")
#                 self.tokenizer = BertTokenizer.from_pretrained(
#                     # 'google-bert/bert-base-uncased',
#                     '/root/autodl-tmp/softprompt/models/bert-large-uncased',
#                     trust_remote_code=True,
#                     local_files_only=False
#                 )
#         elif configs.llm_model == 'pythia-14m':
#             self.pythia_config = GPTNeoXConfig.from_pretrained('/root/autodl-tmp/softprompt/models/pythia-14m')

#             self.pythia_config.num_hidden_layers = configs.llm_layers
#             self.pythia_config.output_attentions = True
#             self.pythia_config.output_hidden_states = True
#             print('pythia_config=', self.pythia_config)
#             try:
#                 self.llm_model = GPTNeoXForCausalLM.from_pretrained(
#                     '/root/autodl-tmp/softprompt/models/pythia-14m',
#                     trust_remote_code=True,
#                     local_files_only=True,
#                     config=self.pythia_config,
#                 )
#             except EnvironmentError:  # downloads model from HF is not already done
#                 print("Local model files not found. Attempting to download...")
#                 self.llm_model = GPTNeoXForCausalLM.from_pretrained(
#                     '/root/autodl-tmp/softprompt/models/pythia-14m',
#                     trust_remote_code=True,
#                     local_files_only=False,
#                     config=self.pythia_config,
#                 )

#             try:
#                 self.tokenizer = AutoTokenizer.from_pretrained(
#                     '/root/autodl-tmp/softprompt/models/pythia-14m',
#                     trust_remote_code=True,
#                     local_files_only=True
#                 )
#             except EnvironmentError:  # downloads the tokenizer from HF if not already done
#                 print("Local tokenizer files not found. Atempting to download them..")
#                 self.tokenizer = GPTNeoXTokenizer.from_pretrained(
#                     '/root/autodl-tmp/softprompt/models/pythia-14m',
#                     trust_remote_code=True,
#                     local_files_only=False
#                 )
#         else:
#             raise Exception('LLM model is not defined')

#         if self.tokenizer.eos_token: # 结束token end-of-sentence token
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         else:
#             pad_token = '[PAD]'
#             self.tokenizer.add_special_tokens({'pad_token': pad_token}) # 向tokenizer的词汇表中添加新token
#             self.tokenizer.pad_token = pad_token

#         for param in self.llm_model.parameters():
#             param.requires_grad = False # LLM不训练

#         if configs.prompt_domain:
#             self.description = configs.content # PaP的domain部分
#         else:
#             self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

#         self.dropout = nn.Dropout(configs.dropout)

#         self.patch_embedding = PatchEmbedding(
#             configs.d_model, self.patch_len, self.stride, configs.dropout)

#         # 获取llm输入嵌入层的权重，即词汇表中每个词的嵌入向量
#         self.word_embeddings = self.llm_model.get_input_embeddings().weight
#         self.vocab_size = self.word_embeddings.shape[0]
#         self.num_tokens = 1000 # text prototypes？
#         self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

#         self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

#         self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
#         self.head_nf = self.d_ff * self.patch_nums # 所有头的总维度

#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
#                                                  head_dropout=configs.dropout)
#         else:
#             raise NotImplementedError

#         self.normalize_layers = Normalize(configs.enc_in, affine=False)

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out[:, -self.pred_len:, :]
#         return None

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         print('x_enc.shape=', x_enc.shape) # B, T, N   timellm的N为1

#         x_enc = self.normalize_layers(x_enc, 'norm')

#         B, T, N = x_enc.size()
#         x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) # B*N, T, 1

#         min_values = torch.min(x_enc, dim=1)[0] # 沿着某个channels维度
#         max_values = torch.max(x_enc, dim=1)[0]
#         medians = torch.median(x_enc, dim=1).values
#         lags = self.calcute_lags(x_enc)
#         trends = x_enc.diff(dim=1).sum(dim=1) # 1阶差分，再求和得到趋势信息

#         # PaP
#         prompt = []
#         for b in range(x_enc.shape[0]):
#             min_values_str = str(min_values[b].tolist()[0])
#             max_values_str = str(max_values[b].tolist()[0])
#             median_values_str = str(medians[b].tolist()[0])
#             lags_values_str = str(lags[b].tolist())
#             prompt_ = (
#                 f"<|start_prompt|>Dataset description: {self.description}"
#                 f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
#                 "Input statistics: "
#                 f"min value {min_values_str}, "
#                 f"max value {max_values_str}, "
#                 f"median value {median_values_str}, "
#                 f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
#                 f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
#             )

#             prompt.append(prompt_)

#         print('len of prompt=', len(prompt)) # B*N

#         x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous() # B, T, N

#         prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # (B*N, inputs_ids_num))将文本转成llm可以理解的token 
#         print('after tokenizer prompt,shape=', prompt.shape) # (B*N, inputs_ids_num)
#         prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # llm的输入嵌入层的 输出
#         print('prompt_embeddings.shape=', prompt_embeddings.shape) # (batch*N, prompt_token_num(inputs_ids_num), llm_dim)
#         # 总结：将prompt文本先进行tokenize，每个样本对应的文本转变成token，接着在用llm的embedder进行嵌入，这样每个token映射到更低的维度

#         source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # text prototype (1000, llm_dim) 所选择的那些与时序更相关的token
#         print('source_embeddings.shape=', source_embeddings.shape) # (text prototype num_tokens, llm_dim)

#         x_enc = x_enc.permute(0, 2, 1).contiguous() # B,N,T
#         enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
#         print('after patch embedding enc_out.shape=', enc_out.shape) # B*N, patch_num, d_model
#         enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
#         print('after reprogramming layer enc_out.shape=', enc_out.shape) # B*N, patch_num, llm_model
#         llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1) # PaP与reprogrammed embedding拼接
#         print('after concat作为输入llm的embedding, llama_enc_out.shape=', llama_enc_out.shape) # (B*N, patchnum+inputs_ids_num, llm_dim)
#         dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state # 提取最后一层的隐藏状态
#         print('llm model.last_hidden_state.shape=', dec_out.shape) # B*N, patchnum+inputs_ids_num, llm_dim
#         dec_out = dec_out[:, :, :self.d_ff] # 要保留的特征数，前d_ff (B*N, patchnum+inputs_ids_num, d_ff512)

#         dec_out = torch.reshape(
#             dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])) # batch_size , channels , patchnum+inputs_ids_num , d_ff
#         print('after reshape, dec_out.shape=', dec_out.shape)
#         dec_out = dec_out.permute(0, 1, 3, 2).contiguous() # batch_size , channels , d_ff , patchnum+inputs_ids_num
#         print('after permute, dec_out.shape=', dec_out.shape)

#         dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:]) # 用最后patches个数做预测
#         print('after output_projection, dec_out.shape=', dec_out.shape) # B,N,pred_len
#         dec_out = dec_out.permute(0, 2, 1).contiguous() 

#         dec_out = self.normalize_layers(dec_out, 'denorm') # RevIN
        
#         print('finally before return, dec_out.shape=', dec_out.shape) # B, pred_len, N

#         return dec_out

#     def calcute_lags(self, x_enc):
#         # 快速傅里叶变换
#         q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
#         k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
#         res = q_fft * torch.conj(k_fft) # 频域的自相关性
#         corr = torch.fft.irfft(res, dim=-1) # 时域的自相关性
#         mean_value = torch.mean(corr, dim=1) # 自相关性在每个时间步的平均值
#         _, lags = torch.topk(mean_value, self.top_k, dim=-1) # 自相关性最高的 self.top_k 个时间滞后
#         return lags
# 
# class ReprogrammingLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
#         super(ReprogrammingLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)

#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
#         self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
#         self.n_heads = n_heads
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, target_embedding, source_embedding, value_embedding):
#         B, L, _ = target_embedding.shape
#         S, _ = source_embedding.shape
#         H = self.n_heads

#         target_embedding = self.query_projection(target_embedding).view(B, L, H, -1) # time series embedding
#         source_embedding = self.key_projection(source_embedding).view(S, H, -1) # text prototype
#         value_embedding = self.value_projection(value_embedding).view(S, H, -1)

#         out = self.reprogramming(target_embedding, source_embedding, value_embedding)

#         out = out.reshape(B, L, -1)

#         return self.out_projection(out)

#     def reprogramming(self, target_embedding, source_embedding, value_embedding):
#         B, L, H, E = target_embedding.shape

#         scale = 1. / sqrt(E)

#         scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

#         return reprogramming_embedding

class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.llm_dim = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.softprompt_seq_len = configs.softprompt_seq_len
        self.n_layer = configs.llm_layers
        self.n_head = configs.n_heads
        self.n_embd = configs.llm_dim // configs.n_heads
        self.llm_layers = configs.llm_layers
        self.channels = configs.enc_in
        self.inference_paths = configs.inference_paths
        print('Time-LLM.py class model init()')
       
        # print('n_head', self.n_head) # 8
        # print('seq_len', self.seq_len) # 96
        # print('head_dim', self.n_embd) # 1024/8=128
        # print('configs.llm_model=', configs.llm_model)

        if configs.llm_model == 'pythia-14m':
            self.pythia_config = GPTNeoXConfig.from_pretrained('/data/home/zhuomin/project/softprompt/models/pythia-14m')

            self.pythia_config.num_hidden_layers = configs.llm_layers
            self.pythia_config.num_attention_heads = configs.n_heads
            self.pythia_config.output_attentions = True
            self.pythia_config.output_hidden_states = True
            print('pythia_config=', self.pythia_config)
            try:
                self.llm_model = GPTNeoXForCausalLM.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-14m',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.pythia_config,
                )
                # self.llm_model = BertModel(self.bert_config, add_pooling_layer=False)
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPTNeoXForCausalLM.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-14m',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.pythia_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-14m',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPTNeoXTokenizer.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-14m',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'pythia-1b':
            self.pythia_config = GPTNeoXConfig.from_pretrained('/data/home/zhuomin/project/softprompt/models/pythia-1b')

            self.pythia_config.num_hidden_layers = configs.llm_layers
            self.pythia_config.num_attention_heads = configs.n_heads
            self.pythia_config.output_attentions = True
            self.pythia_config.output_hidden_states = True
            print('pythia_config=', self.pythia_config)
            try:
                self.llm_model = GPTNeoXForCausalLM.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-1b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.pythia_config,
                )
                # self.llm_model = BertModel(self.bert_config, add_pooling_layer=False)
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPTNeoXForCausalLM.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-1b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.pythia_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-1b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPTNeoXTokenizer.from_pretrained(
                    '/data/home/zhuomin/project/softprompt/models/pythia-1b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token: # 结束token end-of-sentence token
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token}) # 向tokenizer的词汇表中添加新token
            self.tokenizer.pad_token = pad_token
        

        for param in self.llm_model.parameters():
            param.requires_grad = False # LLM不训练，但是soft prompt参数是更新的
            
        self.prefix_tokens = torch.arange(self.softprompt_seq_len).long() # 代表离散的单词，这里随机初始化单词encoder会对这些单词进行编码
        self.prefix_encoder = PrefixEncoder(configs)
 
        if configs.prompt_domain:
            self.description = configs.content # PaP的domain部分
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)


        # 为了将ts_representation与past_key_values进行拼接，卷积层
        # self.expand_representation_with_conv = nn.Conv1d(
        #     in_channels=configs.d_model,
        #     out_channels=self.llm_layers*2*self.llm_dim,
        #     kernel_size=1,
        #     bias=False
        # )
        # self.expand_representation_with_conv_new = nn.Conv1d(
        #     in_channels=configs.d_model,
        #     out_channels=self.llm_dim,
        #     kernel_size=1,
        #     bias=False
        # )

        # 卷积，拼到hardprompt
        self.conv_concat_hard = nn.Conv1d(
            in_channels=configs.d_model,
            out_channels=self.llm_dim,
            kernel_size=1,
            bias=False
        )
        # moment输入llm
        self.conv_moment = nn.Conv1d(
            in_channels=1024,
            out_channels=self.llm_dim,
            kernel_size=1,
            bias=False
        )
        # cross attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.llm_dim,
            num_heads=1,
            dropout=configs.dropout
        )
        
        # 将最后一层token映射到输出
        # self.head_nf = self.llm_dim * self.seq_len
        # # self.head_nf = self.llm_dim * 129 # 用全部做预测
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #     self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
        #                                          head_dropout=configs.dropout)
        # else:
        #     raise NotImplementedError
        
        self.output_projection2 = nn.Linear(
                configs.llm_dim, configs.c_out, bias=True)
        self.output_projection_moment = nn.Linear(
                configs.llm_dim, configs.pred_len, bias=True)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        
        # 小模型
        # self.ts_model = TimesNet(configs) # 括号里面参数需要与timellm对应
        # self.ts_model = self.ts_model.to('cuda')
        # # 小模型参数
        # checkpoint_path = os.path.join(f'/data/home/zhuomin/project/softprompt/checkpoints_TimesNet/{configs.task_name}_{configs.model_id}/checkpoint.pth')
        # # print('checkpoint_path=', checkpoint_path)
        # if os.path.exists(checkpoint_path):
        #     checkpoint = torch.load(checkpoint_path)  
        #     self.ts_model.load_state_dict(torch.load(checkpoint_path))
        #     print("Loaded TimesNet model from ", checkpoint_path)
        # else:
        #     raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        # # 冻结小模型的所有参数
        # for param in self.ts_model.parameters():
        #     param.requires_grad = False
        # # 确保所有权重和偏置的类型为 BFloat16
        # for param in self.ts_model.parameters():
        #     if param.dtype == torch.float32:
        #         param.data = param.data.to(dtype=torch.bfloat16)
                
        # MOMENT获取通用表征
        self.foundation_model = MOMENTPipeline.from_pretrained(
            "/data/home/zhuomin/project/softprompt/models/MOMENT-1-large", 
            model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode to learn representations
            # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
        )
        self.foundation_model.init()
        # 冻结基础模型的所有参数
        for param in self.foundation_model.parameters():
            param.requires_grad = False
        # print(self.foundation_model)
        
        
        # 计算参数数量
        # llm_model_param = 0
        # for name, param in self.llm_model.named_parameters(): # llm的所有参数
        #     llm_model_param += param.numel()
        # ts_model_param = 0
        # for name, param in self.ts_model.named_parameters(): # llm的所有参数
        #     ts_model_param += param.numel()
        # conv_model_param = 0
        # for name, param in self.expand_representation_with_conv.named_parameters(): # llm的所有参数
        #     conv_model_param += param.numel()
        # all_param = 0
        # for name, param in self.named_parameters(): # 整个模型的参数
        #     all_param += param.numel()
        # total_param = all_param - llm_model_param - ts_model_param
        # print('total param is {}'.format(total_param)) # 1638496

    # soft prompt
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.llm_model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        
        # print('past_key_values.shape=', past_key_values.shape) # (B, softprompt_seq_len, llm_layers * 2 * llm_dim)
        
        # 【用hard_prompt_embeddings的值进行初始化】
        # tmp = past_key_values
        # tmp = tmp[:, :, :self.llm_dim]
        # tmp2 = self.hard_prompt_embeddings
        # # (B, prompt_token, llm_dim)
        # B, prompt_token, llm_dim = self.hard_prompt_embeddings.shape
        # # (B, softprompt_seq_len, llm_layers * 2 * llm_dim)
        # B, softprompt_seq_len, _ = tmp.shape
        # # 计算需要填充的长度
        # delta = softprompt_seq_len - prompt_token
        # # 如果 prompt_token 大于 softprompt_seq_len，截断 x
        # if delta < 0:
        #     tmp2 = tmp2[:, :softprompt_seq_len, :]
        # else:
        #     # 如果 prompt_token 小于 softprompt_seq_len，用 past_key_values 填充
        #     tmp2 = torch.cat([tmp2, tmp[:, :delta, :]], dim=1)
        # tmp2 = tmp2.repeat(1, 1, self.llm_layers * 2)
        # # print('tmp2.shape=', tmp2.shape)
        # past_key_values = tmp2
        
        # 重塑
        past_key_values = past_key_values.view(
            batch_size,
            self.softprompt_seq_len,
            self.n_layer * 2, # n_layer个层，每个层有key_layer和value_layer
            self.n_head*self.n_embd,
        ) # (B, softprompt_seq_len, num_layers*2, num_head*head_dim)
        past_key_values = past_key_values.permute([2, 0, 1, 3]) # (num_layers*2, B, softprompt_seq_len, num_head*head_dim)
        
        # 拼接ts_representation
        concatenated_tensors = []
        for i in range(self.n_layer*2):
            # 拼接 ts_representation 到 past_key_values 的第 i 个张量
            concatenated_tensor = torch.cat((past_key_values[i], self.ts_representation), dim=1)
            concatenated_tensors.append(concatenated_tensor)
        # 将列表中的张量堆叠起来，形成最终的张量
        final_tensor = torch.stack(concatenated_tensors, dim=0) # (num_layers*2, B, ts_representation+"N", num_head*head_dim)
        past_key_values = final_tensor.permute([1, 2, 0, 3])
        
        # 【拼接MOMENT】
        past_key_values = past_key_values.view(
            batch_size,
            self.softprompt_seq_len+self.channels,
            self.n_layer * 2, # n_layer个层，每个层有key_layer和value_layer
            self.n_head,
            self.n_embd
        )
        # 【拼接Timesnet】
        # past_key_values = past_key_values.view(
        #     batch_size,
        #     self.softprompt_seq_len+self.seq_len+self.pred_len,
        #     self.n_layer * 2, # n_layer个层，每个层有key_layer和value_layer
        #     self.n_head,
        #     self.n_embd
        # ) # (B, prefix_len, num_layers*2, num_head, head_dim)
        
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # 在第一个维度上，分割成大小为2的块。这个维度大小是n_layer * 2，所以分割完后有n_layer个，每个层对应一个
        return past_key_values
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
            
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.batch_size = x_enc.shape[0]
        n_vars = x_enc.shape[2]
        # x_enc:    batch_size x seq_len x channel      4,96,321

        x_enc = self.normalize_layers(x_enc, 'norm') # (batch_size , seq_len , channel)

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0] # 沿着某个channels维度
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc) # 快速傅里叶变换 (batch_size, self.top_k)
        trends = x_enc.diff(dim=1).sum(dim=1) # 1阶差分，再求和得到趋势信息

        # hard prompt 所有通道的统计信息作为一个hardprompt
        hard_prompt = []
        for b in range(B):
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: \n"
            )
            for n in range(N):
               min_values_str = str(min_values[b * N + n].tolist()[0])
               max_values_str = str(max_values[b * N + n].tolist()[0])
               median_values_str = str(medians[b * N + n].tolist()[0])
               lags_values_str = str(lags[b * N + n].tolist())
               prompt_ += (
                  f"Channel {n+1}:\n"
                  f"min value {min_values_str}, "
                  f"max value {max_values_str}, "
                  f"median value {median_values_str}, "
                  f"the trend of input is {'upward' if trends[b * N + n] > 0 else 'downward'}, "
                  f"top 5 lags are : {lags_values_str}\n"
                   )
            prompt_ += "|end_prompt|>"
            hard_prompt.append(prompt_)
        # print('len(hard_prompt)=', len(hard_prompt)) # B*N
        # print('hard_prompt[0]=', hard_prompt[0]) # 第一个样本的第一个维度 的 统计信息

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous() # B, T, N

        hard_prompt = self.tokenizer(hard_prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).input_ids # (B*N, inputs_ids_num) 文本->token
        hard_prompt_embeddings = self.llm_model.get_input_embeddings()(hard_prompt.to(x_enc.device))  # (B*N, prompt_token_num, llm_dim) llm的输入嵌入层 (1283,129,128)
        # print('hard_prompt_embeddings.shape=', hard_prompt_embeddings.shape) # B, prompt_token, llm_dim   每个token映射到更低维度
        self.hard_prompt_embeddings = hard_prompt_embeddings
        
        # 【TimesNet】获取小模型输出的表征
        # if self.task_name == "short_term_forecast":
        #     _, self.ts_representation = self.ts_model(x_enc.to(torch.bfloat16), None, x_dec.to(torch.bfloat16), None)
        # else:
        #     if x_mark_enc== None:
        #         _, self.ts_representation = self.ts_model(x_enc.to(torch.bfloat16), None, x_dec.to(torch.bfloat16), None)
        #     else:
        #         _, self.ts_representation = self.ts_model(x_enc.to(torch.bfloat16), x_mark_enc.to(torch.bfloat16), 
        #                                                   x_dec.to(torch.bfloat16), x_mark_dec.to(torch.bfloat16)) # batch_size, seq_len+pred_len, d_model(256)
        # tmp = self.ts_representation # for cross attention
        
        # 【MOMENT】
        x_enc = x_enc.permute(0, 2, 1).contiguous() # (B,N,T)
        # 通用表征
        if self.task_name == "short_term_forecast":
            self.ts_representation = self.foundation_model(x_enc.to(torch.bfloat16), None, x_dec.to(torch.bfloat16), None)
        else:
            B, N, T = x_enc.shape
            # 初始化一个列表来存储每个通道的输出
            fm_outputs = []
            for i in range(N):
                # 提取第 i 个通道，形状为 (B, 1, T)
                channel = x_enc[:, i:i+1, :]  # 保持形状为 (B, 1, T)
                # 将通道输入到 moment，得到形状为 (B, fm_dim) 的输出
                fm_output = self.foundation_model(x_enc=channel.to(torch.bfloat16)).embeddings  # 假设 foundation_model 的输出形状为 (B, fm_dim)
                fm_outputs.append(fm_output)

            # 将所有通道的输出在 N 维度上拼接，得到形状为 (B, N, fm_dim)
            self.ts_representation = torch.stack(fm_outputs, dim=1)  # B,N,fm_dim

        # 打印 TimeseriesOutputs 对象的所有属性
        # print(dir(self.ts_representation))
        # print('ts_repre.shape=', self.ts_representation.shape)
        tmp = self.ts_representation # for cross attention
        self.ts_representation = self.ts_representation.permute(0,2,1) # B,fm_dim,N
        self.ts_representation = self.conv_moment(self.ts_representation) # batch_size, llm_dim, N
        # print('after conv_moment, ts_repre.shape=', self.ts_representation.shape)
        self.ts_representation = self.ts_representation.permute(0,2,1) # batch_size, N, llm_dim
        x_enc = x_enc.permute(0, 2, 1).contiguous() # (B,T,N)
        
        

        # 【1 TimesNet】卷积方法适配,对表征进行转换，以拼接到past_key_values,B
        # self.ts_representation = self.ts_representation.permute(0, 2, 1) 
        # expanded_representation = self.expand_representation_with_conv(self.ts_representation)  # [batch_size, target_dim, seq_len+pred_len], target_dim=llmlayer*2*llmdim
        # expanded_representation = expanded_representation.view(
        #     self.batch_size, self.llm_layers*2*self.llm_dim, self.seq_len+self.pred_len
        # )  # [batch_size, target_dim, seq_len_pred_len]
        # expanded_representation = expanded_representation.permute(0, 2, 1)  # [batch_size, seq_len+pred_len, target_dim]
        # self.ts_representation = expanded_representation
        
        # 【1 TimesNet】卷积方法适配,对表征进行转换，以拼接到past_key_values,新写法--在ETTh2的96上与旧写法没什么区别
        # self.ts_representation = self.ts_representation.permute(0, 2, 1) 
        # expanded_representation = self.expand_representation_with_conv_new(self.ts_representation)  # [batch_size, llm_dim, seq_len+pred_len],
        # expanded_representation = expanded_representation.permute(0, 2, 1)  # [batch_size, seq_len+pred_len, target_dim]
        # self.ts_representation = expanded_representation
        

        # 【5】cross attention融合hard和ts
        # tmp = tmp.permute(0, 2, 1) # batch_size, d_model, seq_len+pred_len
        # expanded_representation = self.conv_concat_hard(tmp) # batch_size, llm_dim, seq_len+pred_len
        # expanded_representation = expanded_representation.permute(0, 2, 1)
        # ts_input = expanded_representation
        # query = hard_prompt_embeddings
        # query = query.permute(1, 0, 2)
        # key = expanded_representation.permute(1, 0, 2)
        # value = key
        # attention_output, weights = self.attention(key, query, query)
        # attention_output = attention_output.permute(1, 0, 2)

        x_enc = x_enc.permute(0, 2, 1).contiguous() # (B,N,T)
        
        batch_size = x_enc.shape[0]
        channels = x_enc.shape[1]
        past_key_values = self.get_prompt(batch_size=batch_size)
        
        
        # print('past_key_values[0].shape', past_key_values[0].shape) # 第一层layer的key和value 
        # print('past_key_values[0][0].shape', past_key_values[0][0].shape) # 第一层要拼接的key_layer
        
        outputs = self.llm_model(
            # input_ids,
            # attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=hard_prompt_embeddings,
            # inputs_embeds=attention_output,
            # inputs_embeds=ts_input,
            inputs_embeds=self.ts_representation, # moment
            # output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=False,
            past_key_values=past_key_values,
            # past_key_values=None,
        )
        # print('outputs type=', type(outputs))  # tuple 0:last_hidden_state, 1:pooler_output, 2:hidden_states, 3:attentions
        
        llm_output_hidden_states = outputs[2] # 取所有hidden_states,  len=llm_layers
        llm_output_last_hidden_state = llm_output_hidden_states[-1] # 最后一层 B, seq, llm_dim
        # print('llm_output_last_hidden_state=', llm_output_last_hidden_state.shape) # 4,7,128
        
        llm_output_last_hidden_state = self.dropout(llm_output_last_hidden_state)

        # output = self.output_projection2(llm_output_last_hidden_state[:,  :, :]) 

        output = self.output_projection_moment(llm_output_last_hidden_state[:,  :, :])
        output = output.permute(0,2,1)
        
        output = self.normalize_layers(output, 'denorm') # RevIN

        return output

    def calcute_lags(self, x_enc):
        # 快速傅里叶变换
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft) # 频域的自相关性
        corr = torch.fft.irfft(res, dim=-1) # 时域的自相关性
        mean_value = torch.mean(corr, dim=1) # 自相关性在每个时间步的平均值
        _, lags = torch.topk(mean_value, self.top_k, dim=-1) # 自相关性最高的 self.top_k 个时间滞后
        return lags
    


# class ReprogrammingLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
#         super(ReprogrammingLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)

#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
#         self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
#         self.n_heads = n_heads
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, target_embedding, source_embedding, value_embedding):
#         B, L, _ = target_embedding.shape
#         S, _ = source_embedding.shape
#         H = self.n_heads

#         target_embedding = self.query_projection(target_embedding).view(B, L, H, -1) # time series embedding
#         source_embedding = self.key_projection(source_embedding).view(S, H, -1) # text prototype
#         value_embedding = self.value_projection(value_embedding).view(S, H, -1)

#         out = self.reprogramming(target_embedding, source_embedding, value_embedding)

#         out = out.reshape(B, L, -1)

#         return self.out_projection(out)

#     def reprogramming(self, target_embedding, source_embedding, value_embedding):
#         B, L, H, E = target_embedding.shape

#         scale = 1. / sqrt(E)

#         scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

#         return reprogramming_embedding

