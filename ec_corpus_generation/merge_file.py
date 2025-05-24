import shutil
import json
import os
from PIL import Image
from io import BytesIO
import base64
import torchvision
from torchvision import models, transforms
import torch
import torch.nn as nn
import argparse
import requests
from urllib import request

def checkRecordSize(fpath):
    lst = []
    fi = open(fpath, "r")
    for line in fi:
        lst.append(line)
    # data = json.loads(json.dumps(fi))
    # # data = json.load(fi)
    # print("Length of dataset: ", len(data))
    fi.close()
    print(len(lst))

def repair_pretrain_vre(in_split, outfile):
    print("Beginning split -> pretraining data conversion")
    fi = open(in_split, "r")
    fo = open(outfile, "w")
    task_name = "visual_grounding"
    count = 0
    for line in fi:
        count += 1
        sid, img, caption, question, answer, gto, dsetname, task_type = line.split("\t")
        img = img.strip()
        task_type = task_type.strip()
        if task_type == "caption":
            pline = f"{sid}\t{img}\t{caption}\t\t\t\t{dsetname}\t{task_type}\n"
        if task_type == "qa":
            pline = f"{sid}\t{img}\t\t{question}\t{answer}\t\t{dsetname}\t{task_type}\n"
        if task_type == "visual grounding":
            pline = f"{sid}\t{img}\t{caption}\t\t{gto}\t\t{dsetname}\t{task_name}\n"

        fo.write(pline)
        if count % 2000 == 0:
            print(f"Converted {count} lines from original data to pretrain format")
    fo.close()
    fi.close()

def get_length(fpath):
    count = 0
    with open(fpath, "r") as f:
        for line in f:
            count += 1
    return count

def length_test(megerged_file, original_file_lst):
    print(get_length(megerged_file))
    for file in original_file_lst:
        print(get_length(file))

def examine_pretrain_set(fpath):
    print("Beiginning examining data")
    i = 0
    data = []
    with open(fpath, "r") as f:
        for line in f:
            data.append(line.split("\t"))
    print("Length:", len(data[0]), "id:", 0, data[0][2:])
    print("Length:", len(data[1]), "id:", 1, data[1][2:])
    print("Length:", len(data[120623]), "id:", 120623, data[120623][2:])
    print("Length:", len(data[120624]), "id:", 120624, data[120624][2:])
    print("Length:", len(data[120625]), "id:", 120625, data[120625][2:])
    print("Length:", len(data[-1]), "id:", -1, data[-1][2:])
    print("Length:", len(data[-2]), "id:", -2, data[-2][2:])

def combine_files(outfile, in_file_lst):
    print("Beginning combining files")
    with open(outfile,'wb') as wfd:
        for f in in_file_lst:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)  
    print("End of file combination") 

in_split = "/home/grads/mogunleye/Research/EC_pretraining/vision_language_examples.tsv"
outfile_caption = "/home/grads/mogunleye/Research/EC_pretraining/test_folder/caption_test.tsv"
outfile_vre = "/home/grads/mogunleye/Research/EC_pretraining/test_folder/vre_test.tsv"
outfile_vre1 = "/home/grads/mogunleye/Research/EC_pretraining/test_folder/vre_test1.tsv"

#miniGPT
in_file_cc = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/ccs_synthetic_filtered_large.json"
in_test = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/test_file_10k.json"
out_test_ec = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/test_file_10k_ec1.json"
out_file_cc1 = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/ccs_synthetic_filtered_large_EC.json"
out_file_cc = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/test_file_10k_ec.json"

#Natural language
general_outfile = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/natural_language/full_pretrain_vre_caption_vqa.tsv"
general_outfile_new = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/natural_language/full_pretrain_vre_caption_vqa_new.tsv"
vre = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/natural_language/refcoco_pretrain_NL.tsv"
caption = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/natural_language/caption_stage1_train_NL.tsv"
vqa = "/data/datasets/VQA/ofa/vqa_data_ofa/pretrain_data/vqa_nl_pretrain_file.tsv"
# combine_files(general_outfile, [vre, caption, vqa])

#EC language
general_outfile_ec = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/ec_language/EC_full_pretrain_vre_caption_vqa.tsv"
general_outfile_ec_new = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/ec_language/EC_full_pretrain_vre_caption_vqa_new.tsv"
vre_ec = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/ec_language/refcoco_pretrain_EC.tsv"
caption_ec = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/ec_language/caption_stage1_train_EC_new.tsv"
vqa_ec = "/data/datasets/VQA/ofa/vqa_data_ofa/pretrain_data/vqa_ec_pretrain_file_vs4035_sl15.tsv"
# combine_files(general_outfile_ec, [vre_ec, caption_ec, vqa_ec])

#reparing pretrain data
# repair_pretrain_vre(general_outfile, general_outfile_new)
# repair_pretrain_vre(general_outfile_ec, general_outfile_ec_new)

#Testing
# length_test(general_outfile_ec, [general_outfile_ec_new])
# length_test(general_outfile_ec, [vre_ec, caption_ec, vqa_ec])
#"/data/datasets/EC_pretraining/OFA/dataset/old_pretrain_data/vision_language_examples_NL.tsv"
# examine_pretrain_set(general_outfile_new)
checkRecordSize(out_file_cc1)
# examine_pretrain_set(caption_ec)
# examine_pretrain_set(vre_ec)
# examine_pretrain_set(vqa_ec)



