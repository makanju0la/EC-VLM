
import os
import argparse
import json

def score_eval(args):
    assert os.path.exists(args.answers)
    assert os.path.exists(args.results)

    fa = open(args.answers, "r")
    fr = open(args.results, "r")
    answers = fa.readlines()
    results = json.loads(fr.read())
    fa.close()
    fr.close()

    answer_dict = {}
    for answer in answers:
        parts = answer.split("\t")
        # answer_dict[int(parts[0])] = answer
        if int(parts[0]) in answer_dict.keys():
            print(f"replacing key {int(parts[0])}")
        if int(parts[0]) == 66046:
            print(f"qid:{parts[0]}, iid:{parts[1]}, q:{parts[2]}")
        answer_dict[int(parts[0])] = parts[3].split("|!+")[1]
    for i, result in enumerate(results):
        print(f"id={result['question_id']}: {result['answer']}, {answer_dict[result['question_id']]}")
        if (i+1)%10==0:
            break
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score evaluation results.')

    # All following args are from previous file (train.py)
    # main ones to change
    parser.add_argument("--results", type=str, help="OFA results file (.json).", required=True)
    parser.add_argument("--answers", type=str, help="Ground truth answers file.", required=True)

    args, remaining_args = parser.parse_known_args()

    # Ex.
    # python score_eval.py --results /home/cdvickery/Research/EC_Pretraining/vqa/ofa/OFA/run_scripts/vqa/vqa_eval_nlfinetune_beamsearch/vqa_val_beam/val_predict.json --answers /projects/virtual_presenter/veqa/vqa_data_ofa/vqa_val.tsv
    score_eval(args)