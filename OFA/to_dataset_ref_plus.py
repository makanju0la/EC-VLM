from models import SingleAgent
import torchvision
from torchvision import models, transforms
import torch
import torch.nn as nn
import argparse
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import time
from PIL import Image
from io import BytesIO
import base64
from pathlib import Path

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

# https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset from images.')

    parser.add_argument('--split', help='Split name ("train" or "val")')
    parser.add_argument('--image_folder', help='Folder that contains split folders containing images.')
    parser.add_argument('--annotation_folder', help='Folder containing annotation files for each split')

    # All following args are from previous file (train.py)
    # main ones to change
    parser.add_argument("--dataset", type=str, default="cc", help="Which Image Dataset To Use EC Pretraining")
    parser.add_argument("--vocab_size", type=int, default=4035, help="EC vocab size")
    parser.add_argument("--seq_len", type=int, default=15, help="Max Len")
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

    # Get image encoder and speaker
    img_encoder = get_image_encoder()
    model = SingleAgent(args)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()
    speaker = model.speaker
    print(speaker)

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

    # FOLDER MANAGEMENT
    # img_base_path = args.image_folder
    # annot_base_path = args.annotation_folder

    # split_img_path = f"{img_base_path}/{args.split}2014"
    # split_anno_path = f"{annot_base_path}/captions_{args.split}2014.json"

    # dataset_path = "./translation_dataset/"
    # Path(dataset_path).mkdir(parents=True, exist_ok=True)

    # speaker_path = dataset_path + f"vocab_{str(args.vocab_size)}_seq_{str(args.seq_len)}/"
    # Path(speaker_path).mkdir(parents=True, exist_ok=True)

    # split_path = speaker_path + f"{str(args.split)}/"
    # Path(split_path).mkdir(parents=True, exist_ok=True)

    # # LOAD COCO DATASET
    # # https://medium.com/howtoai/pytorch-torchvision-coco-dataset-b7f5e8cad82
    # # Might need pycocotools package for this
    # dataset_split = datasets.CocoCaptions(root=split_img_path, annFile=split_anno_path, transform=transform)
    # print(f"Split size: {len(dataset_split)}")
    # data_loader = torch.utils.data.DataLoader(
    #     dataset_split,
    #     batch_size=1,
    # )

    # This is important to disable some non-needed layers that are only needed during training
    img_encoder.eval()
    model.eval()

    #Read in refcoco_pretrain_small.tsv
    # refcoco_path = f"/home/mogunleye/OFA/dataset/pretrain_data/vision_language_examples_NL.tsv"
    refcoco_plus_path = f"/home/mogunleye/OFA/dataset/refcocoplus_data/refcocoplus_train.tsv"
    # refcoco_path = f"/home/mogunleye/ec-nl/data/sample.tsv"
    refcoco_plus = []
    with open(refcoco_plus_path) as f:
        for line in f:
            l=line.split('\t')
            refcoco_plus.append(l)

    with torch.no_grad():
        sample_ct = 0
        start_time = time.time()
        for sample in refcoco_plus:

            # Get refcoco base64 image. 1 for vg pretrain data. 4 for refcoco plus
            base64_str = sample[4]
            # print("last 10 letters in base64 image", base64_str[-10:-1])

            #Transform base64 to image
            img = Image.open(BytesIO(base64.b64decode(base64_str)))
            img = img.convert('RGB')


            # Process through ResNet18 for features
            img_t = transform(img) #transform to format that Resnet was trained on
            batch_t = torch.unsqueeze(img_t, 0) #pass to batch
            features = img_encoder(batch_t)


            # Get EC caption for image (ended up just using model b/c of convenient generate_ec function)
            comm_action = model.generate_ec(features.cuda())
            ec_text = comm_action.tolist()
            ec_text_str = " ".join([str(x) for x in ec_text[0]])
            
            sample[2] = ec_text_str
            if sample_ct % 20 == 0:
                print("EC text: ", ec_text_str)
                print(sample)
                print()
            # Make dict for sample
            # sample_dict = {
            #     "id_ct": sample_ct,             # Just an ID number
            #     # "image": img,
            #     # "features": features,
            #     "nl_captions": target,          # Has all captions associated with image
            #     "ec_caption": comm_action,      # Single EC caption generated for image
            # }

            # new_dataset.append(sample_dict)

            # # Save sample for later use (in translator?)
            # fname = f"{split_path}nl_ec-sample_{str(sample_ct).rjust(8, '0')}.pt"
            # torch.save(sample_dict, fname)

            sample_ct += 1
            if sample_ct % 200 == 0:
                frac_complete = sample_ct / len(refcoco_plus)
                curr_time = time.time()
                diff = curr_time - start_time
                est_time_left = (diff / frac_complete) - diff
                print(f"Processed {sample_ct}/{len(refcoco_plus)} images. Done in ~{est_time_left:.1f}s.")

        # Save list as dataset for later use (in translator?)
        # fname = f"{split_path}translation_dataset.pt"
        # torch.save(new_dataset, fname)
    #write sample pretrain_list
    refcoco_write_path = f"/home/mogunleye/OFA/dataset/refcocoplus_data/refcocoplus_train_EC.tsv"
    with open(refcoco_write_path, 'w') as file:
        file.writelines(
            '\t'.join(lst) for lst in refcoco_plus
        )
    