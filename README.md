<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Prompt Learning-based LLM for Time-series Forecasting </b></h2>
  <h2><b> 基于大模型提示学习的时序数据预测方法研究 </b></h2>
</div>

<div align="center">


</div>

</div>

<p align="center">


</p>

---

## Framework
![](./figures/framework.png 'framework')

## Requirements
Use python 3.11 from MiniConda

- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0
- sktime
- openpyxl
- matplotlib
- mpi4py
- momentfm

To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)  or [[Baidu Drive]](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy), then place the downloaded contents under `./dataset`

## Quick Demos
1. Prepare a conda environment. `conda create -n softprompt python=3.11`
2. Activate the conda environment. `conda activate softprompt`
3. Install all dependencies. `pip install -r requirements.txt`
4. Download datasets and place them under `./dataset`
5. Download LLM models from hugging face e.g [[pythia-14m]](https://huggingface.co/EleutherAI/pythia-14m) and place it under `./models`
6. We provide experiment scripts under the folder `./scripts`. For example, you can evaluate on ETT datasets by:

```bash
bash ./scripts/TimeLLM_ETTh2.sh 
```


## Detailed usage

Please refer to ```run_main.py```, ```run_m4.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.
