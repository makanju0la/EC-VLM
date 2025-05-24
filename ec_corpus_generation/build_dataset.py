
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
import json

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

def img_string_to_img_from_url(img_url):
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.convert('RGB')
    return img

def comm_action_to_string(ca):
    ec_caption = " ".join([str(token) for token in ca.squeeze(0).tolist()])
    return ec_caption

def build_pretraining_dataset_from_llava(in_split, outfile, args):
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

    print("Beginning split ->  data conversion (ec)")
    
    image_folder = "/home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/images-595k-cc3m"
    with open(in_split, 'r') as file:
        json_data = json.load(file)
    
    count = 0
    file_length = 595375
    for item in json_data:  # Iterate through each item in the JSON data
        count += 1
        image_file = item['image']
        
        # Create the full image path
        image_path = os.path.join(image_folder, image_file)
        try:
            img = Image.open(image_path).convert('RGB')
        except:
            continue
        else:
            features = img_encoder(transform(img).unsqueeze(0))
            comm_action = model.generate_ec(features.cuda())
            new = comm_action_to_string(comm_action)
            for conversation in item['conversations']:
                if conversation['from'] == 'gpt':
                    conversation['value'] = new
        
        if count % 200 == 0:
            print()
            print(f"Converted {count} lines from caption data to ec pretrain format")
            print(f'{new} at #{count}')
            print(f"{round((count/file_length) *100, 2)}% has been processed; {round((1 - (count/file_length)) * 100, 2)}% remaining")
    with open(outfile, 'w') as ofile:
        json.dump(json_data, ofile, indent=2)

def build_pretraining_dataset_from_llava1_5(in_split, outfile, args):
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

    print("Beginning split ->  data conversion (ec)")
    
    image_folder = "/home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/LLaVA-Pretrain/images_558_laoin_cc_sbu"
    file_length = 10000
    with open(in_split, 'r') as file:
        json_data = json.load(file)
    json_data_10k = json_data[:file_length]
    
    print("file length: ", file_length)
    for ind, item in enumerate(json_data_10k):  # Iterate through each item in the JSON data
        image_file = item['image']
        
        # Create the full image path
        image_path = os.path.join(image_folder, image_file)
        try:
            img = Image.open(image_path).convert('RGB')
        except:
            continue
        else:
            features = img_encoder(transform(img).unsqueeze(0))
            comm_action = model.generate_ec(features.cuda())
            new = comm_action_to_string(comm_action)
            for conversation in item['conversations']:
                if conversation['from'] == 'gpt':
                    conversation['value'] = new
        
        if ind % 200 == 0:
            print()
            print(f"Converted {ind} lines from caption data to ec pretrain format")
            print(f'{new} at #{ind}')
            print(f"{round((ind/file_length) *100, 2)}% has been processed; {round((1 - (ind/file_length)) * 100, 2)}% remaining")
    with open(outfile, 'w') as ofile:
        json.dump(json_data_10k, ofile, indent=2)

def build_pretraining_dataset_from_COCO(outfile, args):
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

    print("Beginning split ->  data conversion (ec)")
    
    image_folder = "/home/grads/mogunleye/Research/EC_pretraining/EC_data/train2014"
    images = os.listdir(image_folder)
    
    file_length = 10000
    #refcoco first 10,000 ec tokens generated by speaker agent with sequence length 5, 15, 25. Generated from the original COCO dataset (train2014)
    with open(outfile, "w") as fout: 
        for i in range(file_length):
            ind_image_file = images[i]
            # Create the full image path
            ind_image_path = os.path.join(image_folder, ind_image_file)
            try:
                img = Image.open(ind_image_path).convert('RGB')
            except:
                continue
            else:
                features = img_encoder(transform(img).unsqueeze(0))
                comm_action = model.generate_ec(features.cuda())
                new = comm_action_to_string(comm_action)
                fout.write(new + "\n")
            
            if i % 200 == 0:
                print()
                print(f"Converted {i} lines from caption data to ec pretrain format")
                print(f'{new} at #{i}')
                print(f"{round((i/file_length) *100, 2)}% has been processed; {round((1 - (i/file_length)) * 100, 2)}% remaining")

def build_pretraining_dataset_VQA_EC_Unigram(in_split, outfile, args):
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

    print("Beginning split -> pretraining VQA data conversion (ec)")
    fi = open(in_split, "r")
    file_length = 10000 #VQA EC data first 10,000 ec tokens generated by speaker agent with sequence length 5, 15, 25.
    with open(outfile, "w") as fo:
        for i, line in enumerate(fi):
            if i >= file_length:
                break
            uid, imgid, question, answer, preds, img_str, num = line.split("\t")
            img_str = img_str.strip()
            img = img_string_to_img(img_str)
            # Process through ResNet18 for features
            features = img_encoder(transform(img).unsqueeze(0))
            # Get EC caption for image (ended up just using model b/c of convenient generate_ec function)
            comm_action = model.generate_ec(features.cuda())
            new = comm_action_to_string(comm_action)
            fo.write(new + "\n")

            if i % 200 == 0:
                print()
                print(f"Converted {i} lines from vqa data to ec pretrain format")
                print(f'{new} at #{i}')
                print(f"{round((i/file_length) *100, 2)}% has been processed; {round((1 - (i/file_length)) * 100, 2)}% remaining")
    fi.close()

def build_pretraining_dataset_from_cc_Sbu_folder(in_split, outfile, args):
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

    print("Beginning split ->  data conversion (ec)")
    
    json_lst = []
    count = 0
    folder_dir_lst = os.listdir(in_split)
    for file in folder_dir_lst:
        count += 1
        current_path = os.path.join(in_split, file)
        try:
            img = Image.open(current_path).convert('RGB')
        except:
            continue
        else:
            features = img_encoder(transform(img).unsqueeze(0))
            comm_action = model.generate_ec(features.cuda())
            new = comm_action_to_string(comm_action)
            img_dict = {"image_id": file[:-4], "caption": new}
            json_lst.append(img_dict)

        if count % 100 == 0:
            print()
            print(f"Converted {count} lines from caption data to ec pretrain format")
            print(f"{round((count/len(folder_dir_lst)) *100, 2)}% has been processed; {round((1 - (count/len(folder_dir_lst))) * 100, 2)}% remaining")

    json_output = {"annotations": json_lst}
    with open(outfile, 'w') as fo:
        fo.write(str(json_output))


def build_pretraining_dataset_url(in_split, outfile, args):
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
    # fo = open(outfile, 'w')
    data = json.load(fi)
    print("Length of dataset: ", len(data))
    fi.close()
    count = 0
    # big_list = []
    with open(outfile, 'w') as fo:
        for i in range(len(data)):
            count += 1
            try:
                res = request.urlopen(data[i]['url']).read()
                img = Image.open(BytesIO(res)).convert('RGB')
                # img = img_string_to_img_from_url(line['url'])
            except:
                continue
            else:
                features = img_encoder(transform(img).unsqueeze(0))
                comm_action = model.generate_ec(features.cuda())
                new = comm_action_to_string(comm_action)
                item = {"caption": new, "url": data[i]['url']}
                fo.write(str(item)+"\n")
                print(i)
            if count % 100 == 0:
                print()
                print(f"Converted {count} lines from caption data to ec pretrain format")
                print(f"{round((count/len(data)) *100, 2)}% has been processed; {round((1 - (count/len(data))) * 100, 2)}% remaining")

    # for line in data:
    #     count += 1
    #     try:
    #         img = img_string_to_img_from_url(line['url'])
    #     except:
    #         continue
    #     # Process through ResNet18 for features
    #     features = img_encoder(transform(img).unsqueeze(0))
    #     # Get EC caption for image (ended up just using model b/c of convenient generate_ec function)
    #     comm_action = model.generate_ec(features.cuda())
    #     new = comm_action_to_string(comm_action)
    #     item = {"caption": new, "url": line['url']}
    #     fo.write(str(item)+"\n")
    #     print(count)
    #     # big_list.append(item)

    #     if count % 100 == 0:
    #         print()
    #         print(f"Converted {count} lines from caption data to ec pretrain format")
    #         print(f"{round((count/len(data)) *100, 2)}% has been processed; {round((1 - (count/len(data))) * 100, 2)}% remaining")
    # fo.close() 

    # Comment out for now
    # with open(outfile, 'w') as fout:   
    #     json.dump(big_list, fout)

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
    dset_name = "refcoco_train"
    task_name = "visual grounding"
    count = 0
    for line in fi:
        count += 1
        uid, imgid, refer_expression, bbox, img_str = line.split("\t")
        img_str = img_str.strip()
        img = img_string_to_img(img_str)
        # Process through ResNet18 for features
        features = img_encoder(transform(img).unsqueeze(0))
        # Get EC caption for image (ended up just using model b/c of convenient generate_ec function)
        comm_action = model.generate_ec(features.cuda())
        new = comm_action_to_string(comm_action)

        # Get string of same format as pretraining data
        pline = f"{uid}\t{img_str}\t{new}\t\t\t{bbox}\t{dset_name}\t{task_name}\n"
        fo.write(pline)
        # break

        if count % 2000 == 0:
            print(f"Converted {count} lines from vqa to ec pretrain format")
    fo.close()
    fi.close()
        

def build_pretraining_dataset_nl(in_split, outfile):
    print("Beginning split -> pretraining data conversion (nl)")
    fi = open(in_split, "r")
    fo = open(outfile, "w")
    dset_name = "refcoco_train"
    task_name = "visual grounding"
    count = 0
    for line in fi:
        count += 1
        uid, imgid, refer_expression, bbox, img_str = line.split("\t")
        img_str = img_str.strip()
        # Get string of same format as pretraining data
        pline = f"{uid}\t{img_str}\t{refer_expression}\t\t\t{bbox}\t{dset_name}\t{task_name}\n"
        fo.write(pline)
        if count % 2000 == 0:
            print(f"Converted {count} lines from refcoco to nl pretrain format")
    fo.close()
    fi.close()

def examine_pretrain_set(fpath):
    with open(fpath, "r") as f:
        for line in f:
            sid, img, caption, question, answer, gto, dsetname, task_type = line.split("\t")
            task_type = task_type.strip()
            if (task_type != "qa"):
                continue
            print(f"id: {sid}")
            print(f"caption: {caption}")
            print(f"question: {question}")
            print(f"answer: {answer}")
            print(f"gto: {gto}")
            print(f"dsetname: {dsetname}")
            break

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
    parser.add_argument("--load", type=str, default="/home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_92.11_800_4035.pt", help="load weights")
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


    # python build_vre_pretrain_dataset.py --vocab_size 4035 --seq_len 15 --load /home/grads/cdvickery/Research/EC_Pretraining/ec-nl/ec-game/ckpt/coco_2014_vocab_4035_seq_15_reset_-1_nlayers_1/run23330/model_95.85_2000_4035.pt
    
    #Ola's files
    # in_split = "/data/datasets/EC_pretraining/OFA/dataset/refcoco/refcoco_train.tsv"
    # outfile_nl = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/natural_language/refcoco_pretrain_NL.tsv"
    # build_pretraining_dataset_nl(in_split, outfile_nl)

    # in_split = "/data/datasets/EC_pretraining/OFA/dataset/refcoco/refcoco_train.tsv"
    # outfile_ec = "/data/datasets/EC_pretraining/OFA/dataset/pretrain/ec_language/refcoco_pretrain_EC.tsv"
    # build_pretraining_dataset_ec(in_split, outfile_ec, args)

    #miniGPT
    # in_file_cc = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/ccs_synthetic_filtered_large.json"
    # in_test = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/test_file_10k.json"
    # out_test_ec = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/test_file_10k_ec1.json"
    # out_file_cc = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/ccs_synthetic_filtered_large_EC.json"

    # in_file_cc1 = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/laion_synthetic_filtered_large.json"
    # out_file_cc1 = "/data/datasets/EC_pretraining/OFA/dataset/minigptv/laion_synthetic_filtered_large_EC.json"
    # build_pretraining_dataset_url(in_file_cc1, out_file_cc1, args)

    #Stage 2 Minigpt
    # in_file = "/data/datasets/OUTPUTS/EC_pretraining/MiniGPT/stage2data/cc_sbu_align/image"
    # out_file = "/data/datasets/OUTPUTS/EC_pretraining/MiniGPT/stage2data/cc_sbu_align/filter_cap_ec.json"
    # build_pretraining_dataset_from_cc_Sbu_folder(in_file, out_file, args)

    #LLaVa training 
    # in_file = "/home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/LLaVA-CC3M-Pretrain-595K-1/chat.json"
    # out_file = "/home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/LLaVA-CC3M-Pretrain-595K-1/chat_ec.json"
    in_file = "/home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
    out_file = "/home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/LLaVA-Pretrain/vocab_size_exp/blip_laion_cc_sbu_558k_ec_vocab_10000.json"
    build_pretraining_dataset_from_llava1_5(in_file, out_file, args)
    
    # python build_dataset.py --vocab_size 10000 --seq_len 15 --load /home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_86.88_800_vocab10000_seq15.pt

    #COCO
    ##Change Outfile, Sequence Length and Agent 
    # outfile = "/data/datasets/EC_pretraining/OFA/dataset/unigram_distribution/refcoco_ec/refcoco_ec_10k_seq_15.txt"
    # build_pretraining_dataset_from_COCO(outfile, args)
    # python build_dataset.py --vocab_size 4035 --seq_len 15 --load /home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_seq15_95.59_900_4035.pt
    # python build_dataset.py --vocab_size 4035 --seq_len 25 --load /home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_seq25_99.27_900_4035.pt
    # python build_dataset.py --vocab_size 4035 --seq_len 5 --load /home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_seq5_76.64_900_4035.pt

    #VQA
    ##Change Outfile, Sequence Length and Agent 
    # insplit = "/data/datasets/EC_pretraining/OFA/dataset/vqa_data/vqa_train_00.tsv"
    # outfile = "/data/datasets/EC_pretraining/OFA/dataset/unigram_distribution/vqa_ec/vqa_ec_10k_seq_5_step_900.txt"
    # build_pretraining_dataset_VQA_EC_Unigram(insplit, outfile, args)
    # python build_dataset.py --vocab_size 4035 --seq_len 15 --load /home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_seq15_95.59_900_4035.pt
    # python build_dataset.py --vocab_size 4035 --seq_len 25 --load /home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_seq25_99.27_900_4035.pt
    # python build_dataset.py --vocab_size 4035 --seq_len 5 --load /home/grads/mogunleye/Research/EC_pretraining/EC_game_ckpt/model_seq5_77.31_800_4035.pt