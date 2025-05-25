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
4. Install to run LLaVA experiments
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

## 🗣️ EC Token Generation via Referential Game

### Step 1: Train the Speaker-Listener Agents

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

### Step 2: Generating EC Corpora
The next step in the EC-Pretraining experiment, after training the referential game, is to convert the natural language of the pretraining dataset of the specific VLM to an EC version. 
Run the following code for the conversion:

```bash
bash convert.sh
```

### Step 3: Pretraining a VLM Backbone with the generated EC corpora.

At this step, pretraining will depend on the VLM backbone. For example, for LLaVA, run the following code to pretrain the LLaVA model on the generated EC VLM corpus:

*LLaVA Pretraining*: https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#train

*LLaVA Finetuning*: https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning

#### General Process for Pretraining: 
- Clone the LLaVA code base
- Convert the pretraining dataset to EC format using the conversion script we provided.
- The conversion can be done after you've trained the EC game above.
- After you have the EC pretraining dataset, you can then go ahead and run the pretraining and finetuning experiment of your VLM backbone. In this case, LLaVA
- Evaluate your experiment following the standard evaluation process if the VLM.
- Then compare results. 



