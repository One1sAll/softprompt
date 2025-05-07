# import torch

# # 检查是否有可用的GPU
# if torch.cuda.is_available():
#     # 创建一个张量
#     x = torch.tensor([1.0, 2.0, 3.0])
#     # 将张量移动到GPU上
#     x = x.cuda()
#     print("张量已移动到GPU上")
#     print(x)
# else:
#     print("没有可用的GPU，张量将在CPU上运行")

# import torch

# # 检查 GPU 是否支持 bf16
# if torch.cuda.is_available():
#     gpu_name = torch.cuda.get_device_name(0)
#     supports_bf16 = torch.cuda.is_bf16_supported()
#     print(f"GPU: {gpu_name}")
#     print(f"Supports bf16: {supports_bf16}")
# else:
#     print("No GPU available.")

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# print(f"Hello from rank {rank} of {size}")



from momentfm import MOMENTPipeline
from pprint import pprint
import torch
import os
import sys

model = MOMENTPipeline.from_pretrained(
    "/data/home/zhuomin/project/softprompt/models/MOMENT-1-large", 
    model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode to learn representations
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)
model.init()
# print(model)

# takes in tensor of shape [batchsize, n_channels, context_length]
x = torch.randn(16, 1, 512)
output = model(x_enc=x)
# pprint(output)
