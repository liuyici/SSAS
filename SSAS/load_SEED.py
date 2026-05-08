import network
from dataloader import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import lr_schedule
import utils
import torch.nn.functional as F
from modules import PseudoLabeledData, load_seed, load_seed_iv, split_data, z_score, normalize, load_seed_for_domain
import numpy as np
import adversarial
from utils import ConditionalEntropyLoss, augment
from models import EMA
from cmd_1 import CMD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import Adver_network
import os


def load_seed(args):#Select Source Domain Algorithm(SSDA)
    """
    Parameters:
        @args: arguments
    通过对抗学习来最大化分布差异性和最小化领域标签的交叉熵损失, 
    从而训练出具有领域差异性的特征，
    再有这些特征对目标域数据分类，
    得到的分类标签的比例即是源域数据的权重矩阵，
    输出经过权重矩阵的源域数据卖给下一阶段的训练做准备。
    
    书写代码的第一步: 将数据处理的过程中标签Y改为领域标签,
    书写代码的第二步：给模型添加梯度反转层，为分布最大化做准备,注意，是在将为过后的特征上添加梯度反转层
    书写代码的第三步：输出训练好的特征,使用test_muda进行分类,
    得到分类标签后,还要添加一个函数Label conversion algorithm(LCA)，这个函数是将分类标签转化成权重矩阵，
    得到新的源域数据后,则是使用MFA_LR模型来进行训练。
    """
    # --------------------------
    # Prepare data
    # --------------------------

    # Load SEED and SEED-IV data
    if args.dataset in ["seed", "seed-iv"]:
        print("DATA:", args.dataset, " SESSION:", args.session)
        # Load imagined speech data
        if args.dataset == "seed":
            X, Y, ture_Y = load_seed_for_domain(args.file_path, session=args.session, feature="de_LDS")
        else:
            # [1 session]
            if args.mixed_sessions == 'per_session':
                X, Y = load_seed_iv(args.file_path, session=args.session)
            # [3 sessions]
            elif args.mixed_sessions == 'mixed':
                X1, Y1 = load_seed_iv(args.file_path, session=1)
                X2, Y2 = load_seed_iv(args.file_path, session=2)
                X3, Y3 = load_seed_iv(args.file_path, session=3)

                X = {}
                Y = {}
                for key in X1.keys():
                    X1[key], _, _ = z_score(X1[key])
                    X2[key], _, _ = z_score(X2[key])
                    X3[key], _, _ = z_score(X3[key])

                    X[key] = np.concatenate((X1[key], X2[key], X3[key]), axis=0)
                    Y[key] = np.concatenate((Y1[key], Y2[key], Y3[key]), axis=0)
            else:
                print("Option [mixed_sessions] is not valid.")
                exit(-1)
    return X, ture_Y