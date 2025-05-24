## EC-VLM: Emergent Corpus Pre-training Benefits Vision Language Models

Our work explores how **emergent communication (EC) tokens**, learned through agent-based referential games, can serve as a scalable and language-free alternative for pretraining vision-language models (VLMs). We hypothesize and show that EC tokens are discrete, interpretable representations grounded in vision and optimized through task-driven communication and can provide an effective substitute where natural language annotations are unavailable or limited.

---

## 🚀 Getting Started

The EC-VLM pipeline consists of three main stages:

1. **Training the EC speaker-listener agents** using a referential game.
2. **Generating EC corpora** to replace or complement natural language captions in VLM pretraining datasets.
3. **Pretraining a VLM Backbone** with this generated EC corpora. 

---

## 🗣️ EC Token Generation via Speaker-Listener Game

### Step 1: Train the Speaker-Listener Agents

Train agents in a referential game using image inputs. The **speaker** encodes images into discrete EC tokens, and the **listener** learns to resolve the correct referent based on these tokens.

- Default configuration:
  - **Token length** = 15
  - **Vocabulary size** = 4035
  - **Dataset** = COCO

> 🧪 We found that this hyperparameter setting provides a good balance between communication success and token entropy, yielding stable and transferable speaker representations.

## Install

To install and reproduce our experiments, it may be recommended to have different conda enviroments for different use cases. For example, the enviroment used to train the Speaker agents is quite different from the one used to train LLaVA-1.5-EC models. 
We have provided some quick started codes below. 

1. Clone this repository and navigate to EC-Pretraining folder
```bash
git clone https://github.com/makanju0la/EC-Pretraining.git
cd EC-Pretraining
```

2. Install package to train Speaker agents
```shell
conda create -n ec-nl python=3.8 -y
conda activate ec-nl
Install pytorch 1.8. Install from https://pytorch.org/get-started/previous-versions/ which contains previous version of pytorch that corresponss with your CUDA version.
pip install scipy==1.4.0 transformers==4.4.2
```
3. Install package to Run OFA experiments
```bash
cd OFA
conda create -n ofa python=3.7.4 -y
conda activate ofa
Install pytorch 1.8.1. Install from https://pytorch.org/get-started/previous-versions/ which contains previous version of pytorch that corresponss with your CUDA version.
pip install torchvision==0.9.1
pip install -r requirements.txt
```
4. Install package to run LLaVA experiments

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
