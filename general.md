### General Process for EC-VLM Pretraining: 
1. Decide on a VLM backbone (e.g OFA). You might also experiment on a Large Vision Language Model like LLaVA or MiniGPT-4. In our paper, we conducted extensive experiments and ablation on LLaVA. 
2. Examine the pretraining dataset of the VLM of choice in prepration for conversion to an EC format. 
3. If working with LLaVA or OFA, we have prepared starter codes for you in our repositoy.
4. Next, convert the pretraining dataset to EC format using the conversion script we provided.
5. The conversion can be done after you've trained the EC referential game described in step 1.
6. After you have the EC pretraining dataset, you can then go ahead and run the pretraining and finetuning experiment of your VLM backbone. For example if working with LLavA, you would run the following, after following the instructions in Step 3 to learn how to properly setup your environment for LLaVA training. 

Pretrain

```bash
cd EC_VLM/LLaVA
bash scripts/v1_5/pretrain.sh
```

Fintune 

```bash
cd EC_VLM/LLaVA
bash scripts/v1_5/finetune_lora.sh
```

7. If you've successfully pretrained and finetuned your EC custom VLM model, evaluation is relatively easy. You just need to follow the standard evaluation process that the the VLM provides. 
8. Last step is to compare the results between models derived after training on pretraining data with natural language with the ones derived after pretraining on EC tokens.