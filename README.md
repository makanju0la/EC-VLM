## EC-VLM: Emergent Corpus Pre-training Benefits Vision Language Models

Our work explores how **emergent communication (EC) tokens**, learned through agent-based referential games, can serve as a scalable and language-free alternative for pretraining vision-language models (VLMs). We demonstrate that EC tokens are discrete, interpretable representations grounded in vision and optimized through task-driven communication and can provide an effective substitute where natural language annotations are unavailable or limited.

---

## Install

To install and reproduce our experiments, we have provided some quick-start codes below. 
Clone this repository and:

1. Install to train Speaker agents
```shell
conda create -n ec-nl python=3.8 -y
conda activate ec-nl
Install pytorch 1.8. 
pip install scipy==1.4.0 transformers==4.4.2
```

2. Install to run OFA experiments
```bash
cd OFA
conda create -n ofa python=3.7.4 -y
conda activate ofa
Install pytorch 1.8.1. 
pip install torchvision==0.9.1
pip install -r requirements.txt
```
3. Install to run LLaVA experiments
*navigate to LLaVA folder*
```bash
cd LLaVA
```
*Install Package*
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

*Install additional packages for training cases*
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
## 🚀 Getting Started

The EC-VLM pipeline consists of three main stages:

1. **Training the EC speaker-listener agents** using a referential game.
2. **Generating EC corpora** to replace or complement natural language captions in VLM pretraining datasets.
3. **Pretraining a VLM Backbone** with this generated EC corpora. 

---

## Step 1: 🗣️ Train the Speaker-Listener Agents

Train agents in a referential game using image inputs. The **speaker** encodes images into discrete EC tokens, and the **listener** learns to resolve the correct referent based on these tokens.

- Default configuration:
  - **Token length** = 15
  - **Vocabulary size** = 4035
  - **Dataset** = COCO

> 🧪 We found that this hyperparameter setting provides a good balance between communication success and token entropy, yielding stable and transferable speaker representations.

**Data**

The data for the Speaker-Listener Agents training can be found here. [Google Drive](https://drive.google.com/drive/folders/1dBdGaZzvQ4yn-RMpDMxLlFNLzSSbkOWF?usp=sharing). This includes:

- ```image_features```: Image features of coco-2014 (``coco.pt``) and Conceptual Captions (``cc.pt``) datasets from a pre-trained ResNet, to be used in EC pre-training.

**Training**

To train the Speaker-Listener Agents
```bash
cd ec-game
python train.py
```
---
## Step 2: Generating EC Corpora
The next step in the EC-Pretraining experiment, after training the referential game, is to convert the natural language of the pretraining dataset of the specific VLM to an EC version. 
Run the following code for the conversion:

```bash
bash convert.sh
```

---
## Step 3: Pretraining a VLM Backbone with the generated EC corpora.

At this step, pretraining will depend on the VLM backbone. For example, for LLaVA, run the following code to pretrain the LLaVA model on the generated EC VLM corpus:

### LLaVA Pretraining
> https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#train


LLaVA training involves two main stages:

1. **Feature Alignment**: Align a *frozen* pretrained vision encoder with a *frozen* LLM using 558K image-text pairs from a subset of LAION-CC-SBU.
2. **Visual Instruction Tuning**: Train the model to follow multimodal instructions using a mix of GPT-generated samples and academic VQA datasets.

> Recommended hardware: 8×A100 GPUs with 80GB memory  
> To train on fewer GPUs, adjust:
> ```
> per_device_train_batch_size × gradient_accumulation_steps × num_gpus = constant global batch size
> ```

---

### Vicuna Checkpoints

The base LLM is [Vicuna v1.5](https://lmsys.org/blog/2023-03-30-vicuna/), an instruction-tuned model.  
This checkpoint will be downloaded automatically when using the provided training scripts.


### Stage 1: Pretraining (Feature Alignment)

- **Dataset**: [LAION-CC-SBU 558K subset (with BLIP captions)](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- **Training Time**:
  - LLaVA-13B: ~5.5 hours (8×A100-80G)
  - LLaVA-7B: ~3.5 hours
- **Resolution**: 336px
- **Script**: 
```bash
cd EC_VLM/LLaVA
bash scripts/v1_5/pretrain.sh
```

### LLaVA Finetuning - Stage 2 
> https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning

#### 1. Download Instruction Tuning Data

- Annotations: [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
- Image datasets:
  - [COCO train2017](http://images.cocodataset.org/zips/train2017.zip)
  - [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
  - [OCR-VQA (script)](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) *(use `.jpg` format)*
  - [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
  - [Visual Genome Part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)  
    [Part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Organize the data under `./playground/data/`

#### 2. Start Instruction Tuning

- Download projectors: [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). You may download pretrained projectors and checkpoints if neccessary. LLaVA recommends avoiding legacy projectors due to code compatibility issues. 
- **Training Time**:
  - LLaVA-13B: ~20 hours (8×A100-80G)
  - LLaVA-7B: ~10 hours (8×A100-40G)
- **Script**: 

```bash
cd EC_VLM/LLaVA
bash scripts/v1_5/pretrain.sh
```

### Low GPU Memory Options
If you have low GPU memory, you can Use *LoRA* (Parameter-Efficient Fine-Tuning)
-  You could run 7B model in ~4 NVIDIA A40G GPUs. Make sure `per_device_train_batch_size*gradient_accumulation_steps` is the same as the provided script for best reproducibility.
- You may replace `zero3.json` with `zero3_offload.json` which offloads some parameters to CPU RAM. This slows down the training speed though.
- Script: 

```bash
cd EC_VLM/LLaVA
bash scripts/v1_5/finetune_lora.sh
```

## 📊 Evaluation

LLaVA v1.5 is evaluated across a diverse set of 12 benchmarks. To ensure reproducibility and consistency with the real-time chat demo, all evaluations use **greedy decoding** rather than beam search.

For full details, see the LLaVA [Evaluation Guide](https://github.com/makanju0la/EC-VLM/blob/main/Evaluation.md).


