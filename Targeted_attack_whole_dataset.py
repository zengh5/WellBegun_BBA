'''
The provided code demonstrates how the proposed initialization strategy work with CGBA-H (2023ICCV)
'''

# Packages from the lib
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as torch_models
import time
import torchvision.datasets as dsets
import random
from matplotlib import pyplot as plt

# User-defined packages
from CGBA import Proposed_attack
from utils import LFAA_GaussianFilter, load_ckpt


# Sample_1000 is from Blackboxbench [2025TPAMI]
def load_imagenet_test_data(test_batch_size=1, folder="../../data/imagenet/Sample_1000"):
    val_dataset = dsets.ImageFolder(folder,
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), ]))
    """"""
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    return val_loader

###############################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda'
###############################################################
model_arc = 'resnet50'     # 'incV3', 'ViT', 'resnet50adv'

attack_method = 'CGBA_H'   # CGBA-H is tailored for targeted attack
dim_reduc_factor = 4
# pair_num = 1000

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Models ##############################################################
if model_arc == 'resnet50':
    net = torch_models.resnet50(pretrained=True)
if model_arc == 'incV3':
    net = torch_models.inception_v3(weights=torch_models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=True)
if model_arc == 'ViT':
    net = torch_models.vit_b_32(weights=torch_models.ViT_B_32_Weights.IMAGENET1K_V1)
if model_arc == 'resnet50adv':
    net = torch_models.resnet50(pretrained=True)
    cp_dir = 'the dir of the robust models'
    net = load_ckpt(net, cp_dir + 'resnet50_l2_eps0.1.ckpt')

net = net.to(device)
net.eval()

test_loader = load_imagenet_test_data()
target_label_list = torch.load('tar_label_1000.pt')       # a pseudo-random target class list for reproducibility
normalize = transforms.Compose([transforms.Normalize(mean=mean, std=std)])

WellBeguns = [0, 1]    # 0: baseline, 1: the proposed
for i_WB in range(2):
    # i_WB = 1
    all_norms = []
    all_queries = []
    WellBegun = WellBeguns[i_WB]

    for imgi, (original_image, label) in enumerate(test_loader):
        print('Attack img index:', imgi)
        if imgi < 0:
            continue

        out_label = torch.argmax(net.forward(normalize(original_image).cuda()).data).item()
        real_label = label.item()
        random_index = target_label_list[imgi]
        if out_label == real_label:
            # ATK_target_xi, ATK_target_yi = None, None
            ATK_target_xi, ATK_target_yi = test_loader.dataset[random_index]
            ATK_target_xi = ATK_target_xi.cuda()
            ATK_target_xi_4d = ATK_target_xi.unsqueeze(dim=0)
            ATK_target_F = torch.argmax(net.forward(normalize(ATK_target_xi_4d)).data).cpu()
            while label.item() == ATK_target_yi or ATK_target_F.item() != ATK_target_yi:
                random_index = random.randint(0, len(test_loader) - 1)
                ATK_target_xi, ATK_target_yi = test_loader.dataset[random_index]
                ATK_target_xi = ATK_target_xi.cuda()
                ATK_target_xi_4d = ATK_target_xi.unsqueeze(dim=0)
                ATK_target_F = torch.argmax(net.forward(normalize(ATK_target_xi_4d)).data).cpu()

            t3 = time.time()
            lb = torch.zeros_like(ATK_target_xi_4d)
            ub = torch.ones_like(ATK_target_xi_4d)

            attack = Proposed_attack(net, original_image.cuda(), mean, std, lb, ub,
                                     dim_reduc_factor=dim_reduc_factor,
                                     tar_img=ATK_target_xi_4d,
                                     attack_method=attack_method,
                                     iteration=61,            # control the maximum queries
                                     well_begun=WellBegun)
            x_adv, n_query, norms = attack.Attack()

            t4 = time.time()
            print(f'##################### End Itetations:  took {t4 - t3:.3f} sec #######################')
            all_norms.append(norms)
            all_queries.append(n_query)

        else:
            print('Already missclassified ... Lets try another one!')
        #######

    norm_array = np.array(all_norms)
    query_array = np.array(all_queries)
    norm_median = np.median(norm_array, 0)
    query_median = np.median(query_array, 0)

    np.savez(
        f'Targeted_results/{attack_method}_Tar_{model_arc}_dimRe_{dim_reduc_factor}_WB_{WellBegun}',
        norm=norm_median,
        quer=query_median,
        all_norms=norm_array,
        all_queries=query_array)
