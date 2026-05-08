import numpy as np
from modules import PseudoLabeledData, load_seed, load_seed_iv, split_data, z_score, normalize, load_seed_for_domain


def Weight_X(X,Y,args):
        trg_subj = args.target - 1
        # Target data
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # subjects
        subject_ids = X.keys()
        num_domains = len(subject_ids)

    
        Vx = Tx
        Vy = Ty

        # Standardize target data
        Tx, m, std = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=std)

        print("Target subject:", trg_subj)
        print("Tx:", Tx.shape, " Ty:", Ty.shape)
        print("Vx:", Vx.shape, " Vy:", Vy.shape)
        print("Num. domains:", num_domains)

        print("Data were succesfully loaded")
        
        
        return weight_X,Y,weight




