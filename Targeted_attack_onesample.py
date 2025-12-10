'''
The code shows only one sample in the case the whole dataset is unavailable
'''

# Packages from the lib
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as torch_models
import time
from matplotlib import pyplot as plt
from PIL import Image

# User-defined packages
from CGBA import Proposed_attack
from utils import LFAA_GaussianFilter, load_ckpt


###############################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda'
###############################################################
model_arc = 'resnet50'     # 'incV3', 'ViT', 'resnet50adv'
# model_arc = 'incV3'     # 'incV3', 'ViT', 'resnet50adv'

attack_method = 'CGBA_H'   # CGBA-H is tailored for targeted attack
dim_reduc_factor = 4

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

trn = transforms.Compose([transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(), ])
x_ori = trn(Image.open('sample_image/ILSVRC2012_val_00037780.JPEG'))
x_tar = trn(Image.open('sample_image/ILSVRC2012_val_00035133.JPEG'))

WellBeguns = [0, 1]    # 0: baseline, 1: the proposed
for i_WB in range(2):
    # i_WB = 1
    all_norms = []
    all_queries = []
    WellBegun = WellBeguns[i_WB]

    original_image = x_ori.unsqueeze(dim=0)
    ATK_target_xi = x_tar.cuda()
    ATK_target_xi_4d = ATK_target_xi.unsqueeze(dim=0)

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

