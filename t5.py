from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer

print('t5.py')
# 加载 T5 模型和分词器
# model_name = "google-t5/t5-base"
# model_name = "/mnt/chenzhm39/Time-LLM/models/t5-base"
model_name = "/mnt/chenzhm39/Time-LLM/models/llama-7b"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
llama_config = LlamaConfig.from_pretrained('/mnt/chenzhm39/Time-LLM/models/llama-7b')
print('2')
# model = T5ForConditionalGeneration.from_pretrained(model_name)
llm_model = LlamaModel.from_pretrained(
                    '/mnt/chenzhm39/Time-LLM/models/llama-7b',
                    trust_remote_code=True, # 信任远程代码
                    local_files_only=True, # 只从本地文件系统加载模型，而不是从 Hugging Face 的模型库中下载。这意味着模型文件应该已经存在于本地文件系统中。
                    config=llama_config,
                    # load_in_4bit=True
                )
print('3')

# 准备输入数据
# statistics = "均值=10，方差=2，最大值=15，最小值=5，趋势=上升。"
# input_text = f"给定以下时序数据统计特征：{statistics} 请生成关于这些特征的描述。"

# print('4')
# # 编码输入
# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# print('5')
# # 生成描述
# num_descriptions = 1
# outputs = model.generate(input_ids, max_length=100, num_return_sequences=num_descriptions, num_beams=50, early_stopping=True)

# print('6')
# # 解码输出
# descriptions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# print('7')
# # 打印生成的描述
# for i, desc in enumerate(descriptions):
#     print(f"描述 {i+1}: {desc}")