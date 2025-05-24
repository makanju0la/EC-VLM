CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python model_vqa.py \
    --model-path /home/grads/mogunleye/Research/LLaVA/checkpoints/llava-vicuna-7b-v1.5-finetune \
    --question-file \
    /home/grads/mogunleye/Research/LLaVA/playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/coco2014_val \
    --answers-file \
    /home/grads/mogunleye/Research/LLaVA/playground/data/coco2014_val_qa_eval/answer-file-our-1.jsonl \
