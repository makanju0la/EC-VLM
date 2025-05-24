import os
import json
path_to_data = "/home/mogunleye/OFA/dataset"
# test = "refcoco_data/refcoco_testA.tsv"
# train = "refcoco_data/refcoco_train.tsv"
ref_plus_train = "refcocoplus_data/refcocoplus_train.tsv"
sample_pretrain = "pretrain_data/vision_language_examples_EC.tsv"

#read Visual grounding EC data
refcocoEC = []
with open(os.path.join(path_to_data, sample_pretrain)) as f:
    for line in f:
        l=line.split('\t')
        refcocoEC.append(l)


print(len(refcocoEC))
print(len(refcocoEC[0]), len(refcocoEC[1]), len(refcocoEC[-1]))
print(refcocoEC[-2][2:])
print((refcocoEC[-4][2:]))
print((refcocoEC[-3][2:]))

#read refcoco plus
refcocoPlus = []
with open(os.path.join(path_to_data, ref_plus_train)) as f:
    for line in f:
        l=line.split('\t')
        refcocoPlus.append(l)

print(len(refcocoPlus), len(refcocoPlus[0]), len(refcocoPlus[1]), len(refcocoPlus[-1]))
print(" ")
print(refcocoPlus[0])
print(" ")
print(refcocoPlus[-0][:4])
print(refcocoPlus[-1][:4])

#write refcoco plus train with EC text
# with open(os.path.join(path_to_data, "refcocoplus_data/refcocoplus_train_EC.tsv"), 'w') as file:
#     file.writelines(
#         '\t'.join(lst) for lst in sample_pretrain_lst
#     )