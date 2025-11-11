# MoCHA
## MoCHA: Advanced Vision-Language Reasoning with MoE Connector and Hierarchical Group Attention

* 2025-7-31 The code is available.

## 💪 Getting Started

### Installation

Create a conda environment:

```python
git clone git@github.com:Pbhgit/MoCHA.git
cd MoCHA
conda create -n mocha python=3.10
conda activate mocha
pip install -e .
```

### 🚀Dataset Preparation

We train MoCHA on two mainstream LLMs, Phi-2 2.7B and Vicuna-7B, using the [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) evaluation setting training data. Please update the data paths in the training scripts to match your local directories.

### 🌋Training

After downloading the datasets and JSON files, you can train the model using the following commands. **Step 1: Pre-train the MoECs**

* Language model: Phi2-2.7B

  ```python
  bash scripts/moec_phi/pretrain.sh 
  ```

* Language model: Vicuna-7B-v1.5

  ```python
  bash scripts/moec_vicuna/pretrain.sh 
  ```

**Step 2: Fine-tune the full model** 

* Language model: Phi2-2.7B

  ```python
  bash scripts/moec_vicuna/finetune.sh
  ```

* Language model: Vicuna-7B-v1.5

  ```python
  bash scripts/moec_vicuna/finetune.sh
  ```

  

### 🎯Evaluation

We evaluate MoCHA models on multiple benchmarks and many scripts are based on [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) evaluation settings. We've adapted some of them into multi-GPU evaluation scripts and added evaluation on Mathvista. 

* ScienceQA

  Multi-gpu inference
  
  ```python
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/sqa.sh
  ```

* TextVQA

  Multi-gpu inference
  
  ```python
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/textvqa.sh
  ```

* POPE

  Multi-gpu inference
  
  ```python
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/pope.sh
  ```

* MME

  Multi-gpu inference

  ```python
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/mme.sh
  ```

* GQA

  Multi-gpu inference

  ```pyhton
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa.sh
  ```

* MM-Vet

  Sigle-gpu inference

  ```python
  CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh
  ```

* MMBench

  Single-gpu inference

  ```python
  CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh
  ```

* MathVista

  Single-gpu inference

  ```python
  CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mathvista.sh
  ```

  Note that we use gpt-3.5-turbo for evaluation and you may specify your own API key for evaluation.

