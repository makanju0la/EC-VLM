
import os
from PIL import Image
from io import BytesIO
import base64
import torchvision
from torchvision import models, transforms
import torch
import torch.nn as nn
import argparse

import matplotlib.pyplot as plt

from models import SingleAgent


# https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
# Get image encoder (ResNet18) as per https://openreview.net/pdf?id=49A1Y6tRhaq
def get_image_encoder() -> torch.nn.Module:
    # img_encoder = models.resnet18(pretrained=True)
    img_encoder = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # Replace last fc layer with Identity layer
    img_encoder.fc = Identity()
    return img_encoder

def img_string_to_img(img_str):
    imgdata = base64.b64decode(img_str)
    img = Image.open(BytesIO(imgdata))
    img = img.convert('RGB')
    return img

def comm_action_to_string(ca):
    ec_caption = " ".join([str(token) for token in ca.squeeze(0).tolist()])
    return ec_caption

def build_pretraining_dataset_ec(in_split, outfile, args):
    print("Getting ResNet encoder...")
    img_encoder = get_image_encoder()
    print("Getting EC speaker")
    model = SingleAgent(args)
    model.load_state_dict(torch.load(args.load))
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()
    print(f"Loaded model {model}")
    # Define image transforms needed for ResNet-18 input
    # According to https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
    # and https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html
    transform = transforms.Compose([            #[1]
        transforms.Resize(256),                    #[2]
        transforms.CenterCrop(224),                #[3]
        transforms.ToTensor(),                     #[4]
        transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],                #[6]
        std=[0.229, 0.224, 0.225]                  #[7]
        )])
    print(f"Making image transform {transform}")

    print("Beginning split -> pretraining data conversion (ec)")
    fi = open(in_split, "r")
    fo = open(outfile, "w")
    dset_name = "cc12m"
    task_name = "caption"
    count = 0
    for line in fi:
        count += 1
        uid, imgid, caption, pred_obj_labels, img_str = line.split("\t")
        img_str = img_str.strip()
        img = img_string_to_img(img_str)
        # Process through ResNet18 for features
        features = img_encoder(transform(img).unsqueeze(0))
        # Get EC caption for image (ended up just using model b/c of convenient generate_ec function)
        comm_action = model.generate_ec(features.cuda())
        new = comm_action_to_string(comm_action)

        # Get string of same format as pretraining data
        pline = f"{uid}\t{img_str}\t{new}\t\t\t\t{dset_name}\t{task_name}\n"
        fo.write(pline)
        # break

        if count % 2000 == 0:
            print(f"Converted {count} lines from refcoco to ec pretrain format")
    fo.close()
    fi.close()
        
#uniq-id, image-id, caption, predicted object labels (taken from VinVL, not used), image base64 string 
def build_pretraining_dataset_nl(in_split, outfile):
    print("Beginning split -> pretraining data conversion (nl)")
    fi = open(in_split, "r")
    fo = open(outfile, "w")
    dset_name = "cc12m"
    task_name = "caption"
    count = 0
    for line in fi:
        count += 1
        uid, imgid, caption, pred_obj_labels, img_str = line.split("\t")
        # sid, img, caption, question, answer, gto, dsetname, task_type = line.split("\t")
        # img = img.strip()
        img_str = img_str.strip()
        # Get string of same format as pretraining data
        pline = f"{uid}\t{img_str}\t{caption}\t\t\t\t{dset_name}\t{task_name}\n"
        fo.write(pline)
        if count % 2000 == 0:
            print(f"Converted {count} lines from caption data to nl pretrain format")
    fo.close()
    fi.close()

#for merging
def build_caption_nl(in_split, outfile):
    print("Beginning split -> pretraining data conversion (nl)")
    fi = open(in_split, "r")
    fo = open(outfile, "w")
    dset_name = "cc12m"
    task_name = "caption"
    count = 0
    for line in fi:
        count += 1
        sid, img, caption, question, answer, gto, dsetname, task_type = line.split("\t")
        img = img.strip()
        # Get string of same format as pretraining data
        pline = f"{sid}\t{img}\t{caption}\t\t\t\t{dset_name}\t{task_name}\n"
        fo.write(pline)
        if count % 2000 == 0:
            print(f"Converted {count} lines from caption data to nl pretrain format")
    fo.close()
    fi.close()

#for merging
def build_vre_nl(in_split, outfile):
    print("Beginning split -> pretraining data conversion (nl)")
    fi = open(in_split, "r")
    fo = open(outfile, "w")
    dset_name = "cc12m"
    task_name = "caption"
    count = 0
    for line in fi:
        count += 1
        sid, img, caption, question, answer, gto, dsetname, task_type = line.split("\t")
        img = img.strip()
        # Get string of same format as pretraining data
        pline = f"{sid}\t{img}\t{caption}\t\t\t{gto}\t{dsetname}\t{task_type.strip()}\n"
        fo.write(pline)
        if count % 2000 == 0:
            print(f"Converted {count} lines from caption data to nl pretrain format")
    fo.close()
    fi.close()

def get_length(fpath):
    count = 0
    with open(fpath, "r") as f:
        for line in f:
            count += 1
    return count
    
def join_pretrain(in_file1, in_file2):
    fi1 = open(in_file1, "r")
    fi2 = open(in_file2, "a")
    for line in fi1:
        fi2.write(line)
    fi1.close()
    fi2.close()


def examine_pretrain_set(fpath):
    print("Beiginning examining data")
    with open(fpath, "r") as f:
        i = 0
        for line in f:
            print("Length:", len(line.split("\t")), "id:", i+1, line.split("\t")[2:])
            i+=1
            # sid, img, caption, question, answer, gto, dsetname, task_type = line.split("\t")
            # task_type = task_type.strip()
            # # i += 1
            # # if (task_type != "caption"):
            # #     continue
            # print(f"id: {sid}")
            # print(f"caption: {caption}")
            # print(f"question: {question}")
            # print(f"answer: {answer}")
            # print(f"gto: {gto}")
            # print(f"dsetname: {dsetname}")
            # print(f"task name: {task_type}")

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Create dataset from images.')

    # parser.add_argument('--split', help='Split name ("train" or "val")', required=True)
    # parser.add_argument('--image_folder', help='Folder that contains split folders containing images.', required=True)
    # parser.add_argument('--annotation_folder', help='Folder containing annotation files for each split', required=True)

    # All following args are from previous file (train.py)
    # main ones to change
    parser.add_argument("--dataset", type=str, default="cc", help="Which Image Dataset To Use EC Pretraining")
    parser.add_argument("--vocab_size", type=int, default=4035, help="EC vocab size", required=True)
    parser.add_argument("--seq_len", type=int, default=15, help="Max Len", required=True)
    parser.add_argument("--save_every", type=int, default=100, help="Save model output.")
    parser.add_argument("--num_games", type=int, default=1000,  help="Total number of batches to train for")
    parser.add_argument("--extract", type=str, default="", help="extract")
    parser.add_argument("--wandb", type=int, default=0, help="use wandb")
    # others
    parser.add_argument("--gpuid", type=int, default=0, help="Which GPU to run")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--two_fc", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size For Training")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Batch size For Validation")
    parser.add_argument("--num_dist", type=int, default=256, help="Number of Distracting Images For Training")
    parser.add_argument("--num_dist_", type=int, default=128, help="Number of Distracting Images For Validation")
    parser.add_argument("--D_img", type=int, default=2048, help="ResNet feature dimensionality")
    parser.add_argument("--D_hid", type=int, default=512, help="Token embedding dimensionality")
    parser.add_argument("--D_emb", type=int, default=256, help="Token embedding dimensionality")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout keep probability")
    parser.add_argument("--temp", type=float, default=1.0, help="Gumbel temperature")
    parser.add_argument("--hard", action="store_true", default=False, help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--TransferH", action="store_true", default=False, help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--print_every", type=int, default=50, help="Save model output.")
    parser.add_argument("--ECemb", type=int, default=5000, help="Set The EC Embedding Size")
    parser.add_argument("--valid_every", type=int, default=100, help="Validate model every k batches")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_directions", type=float, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--unit_norm", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--loss_type", type=str, default="xent")
    parser.add_argument("--fix_spk", action="store_true", default=False)
    parser.add_argument("--fix_bhd", action="store_true", default=False)
    parser.add_argument("--no_share_bhd", action="store_true", default=False)
    parser.add_argument("--sample_how", type=str, default="gumbel")
    parser.add_argument("--load", type=str, default="", help="load weights")
    parser.add_argument("--no_write", action="store_true", default=False)
    parser.add_argument("--no_terminal", action="store_true", default=False)
    parser.add_argument("--reset_lsn", type=int, default=-1, help="reset listener")

    args, remaining_args = parser.parse_known_args()


    # pretrain_example = "/projects/virtual_presenter/veqa/vqa_data_ofa/pretrain_data/vision_language_examples.tsv"
    # pretrain_example = "/projects/virtual_presenter/veqa/vqa_data_ofa/pretrain_data/vqa_nl_pretrain_file.tsv"
    # examine_pretrain_set(pretrain_example)
    
    # in_split = "/projects/virtual_presenter/veqa/vqa_data_ofa/vqa_train_s1.tsv"
    # outfile = "/projects/virtual_presenter/veqa/vqa_data_ofa/pretrain_data/vqa_nl_pretrain_file.tsv"
    # build_pretraining_dataset_nl(in_split, outfile)

    # python build_vqa_pretrain_dataset.py --vocab_size 10000 --seq_len 25 --load /home/cdvickery/Research/EC_Pretraining/ec-nl/ec-game/ckpt/coco_2014_vocab_10000_seq_25_reset_-1_nlayers_1/run29075/model_99.8_2000_10000.pt
    # in_split = "/projects/virtual_presenter/veqa/vqa_data_ofa/vqa_train_s1.tsv"
    # outfile = "/projects/virtual_presenter/veqa/vqa_data_ofa/pretrain_data/vqa_ec_pretrain_file.tsv"
    # build_pretraining_dataset_ec(in_split, outfile, args)


    # python build_caption_pretrain_dataset.py --vocab_size 4035 --seq_len 15 --load /home/grads/cdvickery/Research/EC_Pretraining/ec-nl/ec-game/ckpt/coco_2014_vocab_4035_seq_15_reset_-1_nlayers_1/run23330/model_95.85_2000_4035.pt
    
    #Ola's files
    #Caption NL
    in_split = "/data/datasets/EC_pretraining/OFA/dataset/caption/caption_data/caption_val.tsv"
    in_split2 = "/data/datasets/EC_pretraining/OFA/dataset/caption/caption_data/caption_test.tsv"
    # outfile_nl = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/natural_language/caption_stage1_train_NL.tsv"
    # build_pretraining_dataset_nl(in_split, outfile_nl)

 
    
    #Caption EC
    # in_split = "/data/datasets/EC_pretraining/OFA/dataset/caption/caption_data/caption_stage1_train.tsv"
    # outfile_ec = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/ec_language/caption_stage1_train_EC.tsv"
    # new_outfile_ec = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/ec_language/caption_stage1_train_EC_new.tsv"
    # build_pretraining_dataset_ec(in_split, outfile_ec, args)
    # build_pretraining_dataset_nl(outfile_ec, new_outfile_ec)
    # examine_pretrain_set(new_outfile_ec)
    print(get_length(in_split), get_length(in_split2))

    #merge and test data
    # in_split = "/home/grads/mogunleye/Research/EC_pretraining/vision_language_examples.tsv"
    # outfile_caption = "/home/grads/mogunleye/Research/EC_pretraining/test_folder/caption_test.tsv"
    # outfile_vre = "/home/grads/mogunleye/Research/EC_pretraining/test_folder/vre_test.tsv"
    # outfile_vre1 = "/home/grads/mogunleye/Research/EC_pretraining/test_folder/vre_test1.tsv"
    # build_caption_nl(in_split, outfile_caption)
    # build_vre_nl(in_split, outfile_vre)
    # examine_pretrain_set(outfile_caption)
    # print(get_length(outfile_vre), get_length(outfile_caption))
    
    # join_pretrain(outfile_caption, outfile_vre1)
    # print(get_length(outfile_vre), get_length(outfile_vre1))
    # examine_pretrain_set(outfile_vre1)

    # import shutil
    # with open(outfile_vre1,'wb') as wfd:
    #     for f in [outfile_caption,outfile_vre]:
    #         with open(f,'rb') as fd:
    #             shutil.copyfileobj(fd, wfd)
    # examine_pretrain_set(in_split)
