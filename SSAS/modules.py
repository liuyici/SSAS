import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.io
from torch.utils.data import Dataset
import torch
import math
import matplotlib.pyplot as plt
from sklearn import manifold
from utils import ConditionalEntropyLoss, augment, centroid_A
def load_seed(path, session="all", feature="LDS", n_samples=185):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    """
    
    
    session1 = [
        "1_20131027",
        "2_20140404", 
        "3_20140603", 
        "4_20140621", 
        "5_20140411", 
        "6_20130712", 
        "7_20131027",
        "8_20140511",
        "9_20140620",
        "10_20131130", 
        "11_20140618",
        "12_20131127",
        "13_20140527", 
        "14_20140601", 
        "15_20130709"
        ]
        
    session2 = [
        "1_20131030", 
        "2_20140413", 
        "3_20140611", 
        "4_20140702",
        "5_20140418",  
        "6_20131016", 
        "7_20131030", 
        "8_20140514", 
        "9_20140627", 
        "10_20131204",  
        "11_20140625",
        "12_20131201", 
        "13_20140603", 
        "14_20140615",
        "15_20131016",
        ]
        
    # SESSION 3
    
    session3 = [
        "1_20131107",
        "2_20140419",
        "3_20140629",
        "4_20140705",
        "5_20140506", 
        "6_20131113",
        "7_20131106",
        "8_20140521",
        "9_20140704",
        "10_20131211",
        "11_20140630",
        "12_20131207",
        "13_20140610", 
        "14_20140627",
        "15_20131105"
        ]

        
        
    # LABELS
    labels = scipy.io.loadmat(path + "label.mat", mat_dtype=True)
    y_session = labels["label"][0]
    # relabel to neural networks [0,1,2]
    for i in range(len(y_session)):
        y_session[i] += 1
    print(y_session)
    
    # select session
    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
    subjects = 0
    for subj in x_session:
        # load data .mat
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)
        print("Subject load:", subj)
        
        for i in range(15):
            
            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band
            features = dataMat[feature+str(i+1)]
            # [1D]
            features = np.swapaxes(features, 0, 1)

            # [select last 'n_samples' samples]
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]


            # [Build temporal samples]
            # + ++ + + + + + + + + ++  +
            feats = features
            window_size = 9
            temp_feats = None
            b = False
            for a in range(len(feats) - window_size + 1):
                f = feats[a:a+window_size]
                f = np.expand_dims(f, axis=0)
                if not b:
                    temp_feats = f
                    b = True
                else:
                    temp_feats = np.concatenate((temp_feats, f), axis=0)
            features = temp_feats
            # ++ + ++ + + + + + ++ +

            # set labels for each epoch
            labels = np.array([y_session[i]] * features.shape[0])
            # labels = np.array([subjects * features.shape[0]])
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)#(39825,9,62,5)（样本，时间窗，通道，频带）
                Y = np.concatenate((Y, labels), axis=0)#(39825,1)
        subjects += 1
        if samples_by_subject == 0:
            samples_by_subject = len(X)
        #需要考虑保留频带还是保留时间窗

            

    # # reorder data by subject
    # X_subjects = {}
    # Y_subjects = {}
    # n = samples_by_subject
    # r = 0
    # CA_X = np.zeros((len(X), 9, 62,5))#CA对齐所添加模块
    # X = np.transpose(X, (0, 2, 1, 3))#CA对齐所添加模块
    # for en in range(5):#CA对齐所添加模块
    #     print(en)#CA对齐所添加模块
    #     imp_x = X[:,:,:,en]#CA对齐所添加模块
    #     # imp_xx = np.squeeze(imp_x, axis=1)
    #     CA_X[:,:,:,en] = centroid_A(imp_x).cpu().numpy()#CA对齐所添加模块
        
    # for subj in range(len(x_session)):
    #     X_subjects[subj] = CA_X[r:r+n]#CA对齐所修改模块
    #     Y_subjects[subj] = Y[r:r+n]#CA对齐所修改模块
    #     # increment range
    #     r += n
    #     print(X_subjects[subj].shape)
        # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    
    return X_subjects, Y_subjects
    #         feats = features
    #         window_size = 9
    #         temp_feats = None
    #         b = False
    #         for a in range(len(feats) - window_size + 1):
    #             f = feats[a:a+window_size]
    #             f = np.expand_dims(f, axis=0)
    #             if not b:
    #                 temp_feats = f
    #                 b = True
    #             else:
    #                 temp_feats = np.concatenate((temp_feats, f), axis=0)
    #         features = temp_feats
    #         # ++ + ++ + + + + + ++ +

    #         # set labels for each epoch
    #         labels = np.array([y_session[i]] * features.shape[0])

            
    #         # add to arrays
    #         if flag == 0:
    #             X = features
    #             Y = labels
    #             flag = True
    #         else:
    #             X = np.concatenate((X, features), axis=0)
    #             Y = np.concatenate((Y, labels), axis=0)
        
    #     if samples_by_subject == 0:
    #         samples_by_subject = len(X)
    # # ft = len(Tx)
    # # fv = len(Vx)
    # # print(ft,fv)
    # # new_X = np.zeros((len(X), 5, math.floor(62*(63)/2)))
    # # new_shape = (len(X), 9, 62, 5)
    # # X = np.transpose(X, (0, 2, 1, 3))
    # # for en in range(9):
    # #     imp_x = X[:,en,:,:]
    # #     # imp_xx = np.squeeze(imp_x, axis=1)
    # #     new_X[:,en,:] = augment(imp_x).cpu().numpy()
    # # for en in range(5):
    # #     print(en)
    # #     imp_x = X[:,:,:,en]
    # #     # imp_xx = np.squeeze(imp_x, axis=1)
    # #     new_X[:,en,:] = augment(imp_x).cpu().numpy()
    # # tmp_X = X.reshape(*new_shape)

    # # new_X = augment(tmp_X).cuda()

    # # reorder data by subject
    # X_subjects = {}
    # new_X_subjects = {}
    # Y_subjects = {}
    # n = samples_by_subject
    # r = 0
    # r1 = 0
    # for subj in range(len(x_session)):
    #     X_subjects[subj] = X[r:r+n]
    #     Y_subjects[subj] = Y[r:r+n]
    #     # increment range
    #     r += n
    #     print(X_subjects[subj].shape)
        
    # # for subj1 in range(len(x_session)):
    # #     new_X_subjects[subj1] = new_X[r1:r1+n]
    # #     Y_subjects[subj1] = Y[r1:r1+n]
    # #     # increment range
    # #     r1 += n
    # #     print(new_X_subjects[subj1].shape)
    
    # return new_X_subjects, Y_subjects



def load_seed_iv(dir_name, session="all", n_samples=100):
    """
    SEED IV
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    
    """
    
    # SESSION 1
    session1 = [
        "1_20160518", 
        "2_20150915", 
        "3_20150919", 
        "4_20151111",
        "5_20160406",  
        "6_20150507", 
        "7_20150715", 
        "8_20151103", 
        "9_20151028", 
        "10_20151014",  
        "11_20150916",
        "12_20150725", 
        "13_20151115", 
        "14_20151205",
        "15_20150508"
        ]
    # SESSION 2
    session2 = [
        "1_20161125", 
        "2_20150920", 
        "3_20151018", 
        "4_20151118",
        "5_20160413",  
        "6_20150511", 
        "7_20150717", 
        "8_20151110", 
        "9_20151119", 
        "10_20151021",  
        "11_20150921",
        "12_20150804", 
        "13_20151125", 
        "14_20151208",
        "15_20150514"
        ]
    # SESSION 3
    session3 = [
        "1_20161126",
        "2_20151012",
        "3_20151101",
        "4_20151123",
        "5_20160420",
        "6_20150512", 
        "7_20150721", 
        "8_20151117",
        "9_20151209", 
        "10_20151023",  
        "11_20151011",
        "12_20150807", 
        "13_20161130", 
        "14_20151215",
        "15_20150527"
        ]
    
    # select session
    if session == 1:
        x_session = session1
        y_session = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    elif session == 2:
        x_session = session2
        y_session = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    elif session == 3:
        x_session = session3
        y_session = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
        
    for subj in x_session:
        
        # load data .mat
        dataMat = scipy.io.loadmat(dir_name + str(session) + "/" + subj + ".mat", mat_dtype=True)
        print("Subject load:", subj)

        for i in range(24):
            
            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band

            # [OPTION 1]
            #features = dataMat["de_LDS" + str(i + 1)]

            # [OPTION 2]
            feat1 = dataMat["de_LDS"+str(i+1)]
            feat2 = dataMat["de_movingAve" + str(i + 1)]
            features = np.concatenate((feat1, feat2), axis=2)
            
            # swap frequency bands with epochs
            features = np.swapaxes(features, 0, 1)
            
            # select last samples
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]
            
            # set labels for each epoch
            labels = np.array([y_session[i]]*features.shape[0])
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)
                Y = np.concatenate((Y, labels), axis=0)
        
        if samples_by_subject == 0:
            samples_by_subject = len(X)
    
    # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    
    return X_subjects, Y_subjects


def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z = (mean - X) / (std+0.000000001)

    return z, mean, std

def normalize(X, mean, std):
    z = (mean - X) / (std+0.0000001)
    return z

def one_hot(y, n_cls):
    y_new = []
    y = np.array(y, 'int32')
    for i in range(len(y)):
        target = [0] * n_cls
        target[y[i]] = 1
        y_new.append(target)
    return np.array(y_new, 'int32')

# Obtaining TRAIN and TEST from DATA
def split_data(X, Y, seed, test_size=0.3):

    s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in s.split(X, Y):
        X_tr, X_ts = X[train_index], X[test_index]
        Y_tr, Y_ts = Y[train_index], Y[test_index]

    return X_tr, Y_tr, X_ts, Y_ts



# dataset definition
class PseudoLabeledData(Dataset):
    # load the dataset
    def __init__(self, X, Y, W):
        self.X = torch.Tensor(X).float()
        self.Y = torch.Tensor(Y).long()
        # weights
        self.W = torch.Tensor(W).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx], self.W[idx]]
    

def load_seed_for_domain(path, session="all", feature="LDS", n_samples=185):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    """
    
    
    session1 = [
        "1_20131027",
        "2_20140404", 
        "3_20140603", 
        "4_20140621", 
        "5_20140411", 
        "6_20130712", 
        "7_20131027",
        "8_20140511",
        "9_20140620",
        "10_20131130", 
        "11_20140618",
        "12_20131127",
        "13_20140527", 
        "14_20140601", 
        "15_20130709"
        ]
        
    session2 = [
        "1_20131030", 
        "2_20140413", 
        "3_20140611", 
        "4_20140702",
        "5_20140418",  
        "6_20131016", 
        "7_20131030", 
        "8_20140514", 
        "9_20140627", 
        "10_20131204",  
        "11_20140625",
        "12_20131201", 
        "13_20140603", 
        "14_20140615",
        "15_20131016",
        ]
        
    # SESSION 3
    
    session3 = [
        "1_20131107",
        "2_20140419",
        "3_20140629",
        "4_20140705",
        "5_20140506", 
        "6_20131113",
        "7_20131106",
        "8_20140521",
        "9_20140704",
        "10_20131211",
        "11_20140630",
        "12_20131207",
        "13_20140610", 
        "14_20140627",
        "15_20131105"
        ]

        
        
    # LABELS
    labels = scipy.io.loadmat(path + "label.mat", mat_dtype=True)
    y_session = labels["label"][0]
    # relabel to neural networks [0,1,2]
    for i in range(len(y_session)):
        y_session[i] += 1
    print(y_session)
    
    # select session
    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
    subjects = 0
    for subj in x_session:
        # load data .mat
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)
        print("Subject load:", subj)
        
        for i in range(15):
            
            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band
            features = dataMat[feature+str(i+1)]
            # [1D]
            features = np.swapaxes(features, 0, 1)

            # [select last 'n_samples' samples]
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]


            # [Build temporal samples]
            # + ++ + + + + + + + + ++  +
            feats = features
            window_size = 9
            temp_feats = None
            b = False
            for a in range(len(feats) - window_size + 1):
                f = feats[a:a+window_size]
                f = np.expand_dims(f, axis=0)
                if not b:
                    temp_feats = f
                    b = True
                else:
                    temp_feats = np.concatenate((temp_feats, f), axis=0)
            features = temp_feats
            # ++ + ++ + + + + + ++ +

            # set labels for each epoch
            ture_labels = np.array([y_session[i]] * features.shape[0])
            labels = np.array([[subjects] * features.shape[0]]).T
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                ture_Y = ture_labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)#(39825,9,62,5)（样本，时间窗，通道，频带）
                Y = np.concatenate((Y, labels), axis=0)#(39825,1)
                ture_Y = np.concatenate((ture_Y, ture_labels), axis=0)#(39825,1)
        subjects += 1
        if samples_by_subject == 0:
            samples_by_subject = len(X)
        #需要考虑保留频带还是保留时间窗

            

    # # reorder data by subject
    # X_subjects = {}
    # Y_subjects = {}
    # n = samples_by_subject
    # r = 0
    # CA_X = np.zeros((len(X), 9, 62,5))#CA对齐所添加模块
    # X = np.transpose(X, (0, 2, 1, 3))#CA对齐所添加模块
    # for en in range(5):#CA对齐所添加模块
    #     print(en)#CA对齐所添加模块
    #     imp_x = X[:,:,:,en]#CA对齐所添加模块
    #     # imp_xx = np.squeeze(imp_x, axis=1)
    #     CA_X[:,:,:,en] = centroid_A(imp_x).cpu().numpy()#CA对齐所添加模块
        
    # for subj in range(len(x_session)):
    #     X_subjects[subj] = CA_X[r:r+n]#CA对齐所修改模块
    #     Y_subjects[subj] = Y[r:r+n]#CA对齐所修改模块
    #     # increment range
    #     r += n
    #     print(X_subjects[subj].shape)
        # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    ture_Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        ture_Y_subjects[subj] = ture_Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    
    return X_subjects, Y_subjects, ture_Y_subjects



def load_seed_for_session(path, session="all", feature="LDS", target_num = 1, n_samples=185):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    """
    subject1 = [
        "1_20131027",
        "1_20131030", 
        "1_20131107",
    ]
    subject2 = [
        "2_20140404",
        "2_20140413", 
        "2_20140419",
    ]
    subject3 = [
        "3_20140603", 
        "3_20140611", 
        "3_20140629",
    ]
    subject4 = [
        "4_20140621", 
        "4_20140702", 
        "4_20140705",
    ]
    subject5 = [
        "5_20140411", 
        "5_20140418",  
        "5_20140506", 
    ]
    subject6 = [
        "6_20130712", 
        "6_20131016", 
        "6_20131113",
    ]
    subject7 = [
        "7_20131027",
        "7_20131030", 
        "7_20131106",
    ]
    subject8 = [
        "8_20140511",
        "8_20140514", 
        "8_20140521",
    ]
    subject9 = [
        "9_20140620",
        "9_20140627", 
        "9_20140704",
    ]
    subject10 = [
        "10_20131130", 
        "10_20131204", 
        "10_20131211",
    ]
    subject11 = [
        "11_20140618",
        "11_20140625",
        "11_20140630",
    ]
    subject12 = [
        "12_20131127",
        "12_20131201", 
        "12_20131207",
    ]
    subject13 = [
        "13_20140527", 
        "13_20140603", 
        "13_20140610", 
    ]
    subject14 = [
        "14_20140601", 
        "14_20140615",
        "14_20140627",
    ]
    subject15 = [
        "15_20130709"
        "15_20131016",
        "15_20131105"
    ]
    
    session1 = [
        "1_20131027",
        "2_20140404", 
        "3_20140603", 
        "4_20140621", 
        "5_20140411", 
        "6_20130712", 
        "7_20131027",
        "8_20140511",
        "9_20140620",
        "10_20131130", 
        "11_20140618",
        "12_20131127",
        "13_20140527", 
        "14_20140601", 
        "15_20130709"
        ]
        
    session2 = [
        "1_20131030", 
        "2_20140413", 
        "3_20140611", 
        "4_20140702",
        "5_20140418",  
        "6_20131016", 
        "7_20131030", 
        "8_20140514", 
        "9_20140627", 
        "10_20131204",  
        "11_20140625",
        "12_20131201", 
        "13_20140603", 
        "14_20140615",
        "15_20131016",
        ]
        
    # SESSION 3
    
    session3 = [
        "1_20131107",
        "2_20140419",
        "3_20140629",
        "4_20140705",
        "5_20140506", 
        "6_20131113",
        "7_20131106",
        "8_20140521",
        "9_20140704",
        "10_20131211",
        "11_20140630",
        "12_20131207",
        "13_20140610", 
        "14_20140627",
        "15_20131105"
        ]

        
        
    # LABELS
    labels = scipy.io.loadmat(path + "label.mat", mat_dtype=True)
    y_session = labels["label"][0]
    # relabel to neural networks [0,1,2]
    for i in range(len(y_session)):
        y_session[i] += 1
    print(y_session)
    
    # select session
    if target_num == 1:
        x_session = subject1
    elif target_num == 2:
        x_session = subject2
    elif target_num == 3:
        x_session = subject3
    elif target_num == 4:
        x_session = subject4
    elif target_num == 5:
        x_session = subject5
    elif target_num == 6:
        x_session = subject6
    elif target_num == 7:
        x_session = subject7
    elif target_num == 8:
        x_session = subject8
    elif target_num == 9:
        x_session = subject9
    elif target_num == 10:
        x_session = subject10
    elif target_num == 11:
        x_session = subject11
    elif target_num == 12:
        x_session = subject12
    elif target_num == 13:
        x_session = subject13
    elif target_num == 14:
        x_session = subject14
    elif target_num == 15:
        x_session = subject15
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
    subjects = 0
    for subj in x_session:
        # load data .mat
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)
        print("Subject load:", subj)
        
        for i in range(15):
            
            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band
            features = dataMat[feature+str(i+1)]
            # [1D]
            features = np.swapaxes(features, 0, 1)

            # [select last 'n_samples' samples]
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]


            # [Build temporal samples]
            # + ++ + + + + + + + + ++  +
            feats = features
            window_size = 9
            temp_feats = None
            b = False
            for a in range(len(feats) - window_size + 1):
                f = feats[a:a+window_size]
                f = np.expand_dims(f, axis=0)
                if not b:
                    temp_feats = f
                    b = True
                else:
                    temp_feats = np.concatenate((temp_feats, f), axis=0)
            features = temp_feats
            # ++ + ++ + + + + + ++ +

            # set labels for each epoch
            ture_labels = np.array([y_session[i]] * features.shape[0])
            labels = np.array([[subjects] * features.shape[0]]).T
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                ture_Y = ture_labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)#(39825,9,62,5)（样本，时间窗，通道，频带）
                Y = np.concatenate((Y, labels), axis=0)#(39825,1)
                ture_Y = np.concatenate((ture_Y, ture_labels), axis=0)#(39825,1)
        subjects += 1
        if samples_by_subject == 0:
            samples_by_subject = len(X)
    X_subjects = {}
    Y_subjects = {}
    ture_Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        ture_Y_subjects[subj] = ture_Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    
    return X_subjects, Y_subjects, ture_Y_subjects