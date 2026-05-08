def test_muda(dataset_test, model):
    start_test = True
    features = None
    with torch.no_grad():

        for batch_idx, data in enumerate(dataset_test):
            Tx = data['Tx']
            Ty = data['Ty']
            Tx = Tx.float().cuda()

            # 获得预测结果
            feats, outputs = model(Tx)

            # 200个批次连接
            if start_test:
                all_output = outputs.float().cpu()
                all_label = Ty.float()
                all_target_data = Tx.float()
                features = feats.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, Ty.float()), 0)
                all_target_data = torch.cat((all_target_data, Tx.float()), 0)
                features = np.concatenate((features, feats.float().cpu()), 0)

            # 获得预测标签
        _, predictions = torch.max(all_output, 1)
        # 计算所有样本的acc
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)

        # 计算各种指标
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)

        # f1 = f1_score(y_true, y_pred, average='macro')
        # auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        # matrix = confusion_matrix(y_true, y_pred)

        return accuracy, features, y_pred, all_target_data, all_label, all_output




def generate_masks(tensor, mask_ratio=0.7):
    """
    Generate a different mask for each sample in the batch.

    Args:
    tensor (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width)
    mask_ratio (float): Ratio of values to be masked in each sample. Should be between 0 and 1.

    Returns:
    torch.Tensor: Tensor of masks with the same shape as the input tensor.
    """

    batch_size, num_channels, height, width = tensor.shape
    num_elements = height * width

    # Calculate the number of values to be masked in each sample
    num_values_to_mask = int(num_elements * mask_ratio)

    # Initialize the mask tensor
    masks = torch.ones_like(tensor)

    # Iterate through the batch and create a mask for each sample
    for b in range(batch_size):
        for c in range(num_channels):
            # Generate random indices to mask
            indices_to_mask = torch.randperm(num_elements)[:num_values_to_mask]

            # Convert flat indices to 2D indices
            rows = indices_to_mask // width
            cols = indices_to_mask % width

            # Apply the mask
            masks[b, c, rows, cols] = 0
    masks = masks.view(batch_size, height, num_channels, width)
    return masks    

def generate_channel_masks(tensor, mask_ratio=0.7):
    """
    Generate a mask for each sample in the batch that masks out whole channels.

    Args:
    tensor (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width)
    mask_ratio (float): Ratio of channels to be masked in each sample. Should be between 0 and 1.

    Returns:
    torch.Tensor: Tensor of masks with the same shape as the input tensor.
    """
    batch_size, num_channels, height, width = tensor.shape

    # Calculate the number of channels to mask
    num_channels_to_mask = int(num_channels * mask_ratio)

    # Initialize the mask tensor
    masks = torch.ones_like(tensor)

    # Iterate through the batch and create a mask for each sample
    for b in range(batch_size):
        # Generate random indices to mask channels
        channels_to_mask = torch.randperm(num_channels)[:num_channels_to_mask]

        # Apply the mask to the chosen channels
        masks[b, channels_to_mask, :, :] = 0
    masks = masks.view(batch_size, height, num_channels, width)
    return masks







def CMAE(args):
    """
    Parameters:
        @args: arguments
    """
    # --------------------------
    # 数据导入
    # --------------------------

    # 加载数据集SEED
    criterion = LabelSmooth(num_class=args.num_class).to(args.device)
    if args.dataset in ["seed", "seed-iv"]:
        print("DATA:", args.dataset, " SESSION:", args.session)
        if args.dataset == "seed":
            X, Y, ture_Y = load_seed(args.file_path, session=args.session, feature="de_LDS")
        else:
            # [1 session]
            if args.mixed_sessions == 'per_session':
                X, Y, ture_Y = load_seed_iv(args.file_path, session=args.session)
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

        # 挑选出目标域
        trg_subj = args.target - 1
        #目标域数据
        Tx = np.array(X[trg_subj])
        Ty = np.array(ture_Y[trg_subj])
        # print(Ty)
        # subjects
        subject_ids = X.keys()
        num_domains = len(subject_ids)

        Vx = Tx
        Vy = Ty

      
        Tx, m, std = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=std)


        print("Target subject:", trg_subj)
        print("Tx:", Tx.shape, " Ty:", Ty.shape)
        print("Vx:", Vx.shape, " Vy:", Vy.shape)
        print("Num. domains:", num_domains)

        print("Data were succesfully loaded")

        # Train dataset
        train_loader = UnalignedDataLoader()
        train_loader.initialize(num_domains, X, ture_Y, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
        datasets = train_loader.load_data()
       
        test_loader = UnalignedDataLoaderTesting()
        test_loader.initialize(Vx, Vy, 200, shuffle_testing=False, drop_last_testing=False)
        dataset_test = test_loader.load_data()

    else:
        print("This dataset does not exist.")
        exit(-1)


    # --------------------------
    # Create Deep Neural Network
    # --------------------------
    # For synthetic dataset
    if args.dataset in ["seed", "seed-iv"]:
        # Define Neural Network
        # 2790 for SEED
        # 620 for SEED-IV
        input_size = 3720 if args.dataset == "seed" else 3720   # windows_size=9
        # hidden_size = 310

        model = network.DFN(input_size=input_size, hidden_size=args.hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=args.radius).cuda()
        decoders = [
                    network.Decoder(hidden_size=args.bottleneck_dim, out_dim=input_size).to(args.device)
                      for j in range(num_domains - 1)
                    ]
        adv_net = network.DiscriminatorDANN(in_feature=model.output_num(), radius=10.0, hidden_size=args.bottleneck_dim, max_iter=1000).cuda()
    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    #
    parameter_classifier = [model.get_parameters()[2]]
    # parameter_feature = model.get_parameters()[0:2]
    parameter_feature = model.get_parameters()[0:2] + adv_net.get_parameters()# + decoder.get_parameters() for decoder in decoders
    for k in range(num_domains - 1):
        parameter_feature += decoders[k].get_parameters()     
    optimizer_classifier = torch.optim.SGD(parameter_classifier, lr=args.lr_a, momentum=0.9, weight_decay=0.005)
    optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_a, momentum=0.9, weight_decay=0.005)

    # if gpus are availables
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # ------------------------
    # Model training
    # ------------------------

    # Number of centroids for semantic loss
    if args.dataset in ["seed", "seed-iv"]:
        Cs_memory = []
        for d in range(num_domains):
            Cs_memory.append(torch.zeros(args.num_class, args.bottleneck_dim).cuda())
        Ct_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()

    else:
        print("SETTING number of centroids: The dataset does not exist.")
        exit()





    log_total_loss = []
    my_grl = adversarial.AdversarialLayer()
    my_recon = utils.CosineSimilarityLoss().to(args.device)
    for i in range(args.max_iter1):

        for batch_idx, data in enumerate(datasets):
            # get the source batches
            x_src = list()
            y_src = list()
            d_src = list()
            index = 0

            for domain_idx in range(num_domains - 1):
                tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                # print(tmp_y)
                # labels = torch.from_numpy(np.array([[index] * args.batch_size]).T).type(torch.FloatTensor).flatten().long().cuda()
                x_src.append(tmp_x)
                # d_src.append(labels)
                y_src.append(tmp_y)
            inputs_source = torch.cat(x_src, dim=0)
            # get the target batch
            x_trg = data['Tx'].float().cuda()
            # print(x_trg.shape)
            # Enable model to train
            model.train(True)
            adv_net.train(True)

            # obtain schedule for learning rate
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_a)
            optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_a)

            # Get features target
            features_target, outputs_target = model(x_trg)

            
            rec_loss = 0
            my_recon_loss = 0
            mixSubjectFeature = []
            mixMasks = []
            batch_size, timeWin, num_channels, pindai = x_src[0].shape
            # print(batch_size, timeWin, num_channels, pindai)
            # x_src = x_src.view(batch_size, num_channels, timeWin, width)
            for k in range(num_domains - 1):
                masks = utils.generate_channel_masks(x_src[k].view(batch_size, num_channels, timeWin, pindai), mask_ratio=0.50)
                # print(masks.shape,x_src[k].shape)
                mid_feat, _ = model(x_src[k]*masks)
                rec_loss += utils.marginal(mid_feat, features_target)
                x_out = decoders[k](mid_feat)
                x_out = x_out.view(x_src[k].shape)
                mixSubjectFeature.append(x_out.cuda())
                mixMasks.append(masks.cuda())
                my_recon_loss += my_recon((x_out * (1 - masks)).squeeze(),x_trg.view(x_trg.size(0), -1))

            # for m in range(num_domains - 1):
            #     shared_last_out_2, _ = model(mixSubjectFeature[m])
            #     # x_out = decoders[k](shared_last_out_2)
            #     print((mixSubjectFeature[m] * (1 - mixMasks[m])).squeeze().shape)
            #     rec_loss += utils.marginal((mixSubjectFeature[m] * (1 - mixMasks[m])).view(mixSubjectFeature[m].size(0), -1),x_trg.view(x_trg.size(0), -1))
            rec_loss /= num_domains - 1
            my_recon_loss /= num_domains - 1

     
            pred_src = []
          
            feats = []
            for domain_idx in range(num_domains - 1):
                features_source, outputs_source = model(x_src[domain_idx])
                pred_src.append(outputs_source)
                feats.append(features_source)
        
            feature_outs = torch.cat(feats, dim=0)
            all_features = torch.cat((feature_outs, features_target), dim=0)
            pred_source = torch.cat(pred_src, dim=0)
            labels_source = torch.cat(y_src, dim=0)
            adv_loss = utils.loss_adv(my_grl.apply(all_features), adv_net, logits=torch.nn.Softmax(dim=1)(pred_source).detach())

            # [COARSE-grained training loss]
            classifier_loss = criterion(pred_source, labels_source.flatten())

            # [1] total_loss = classifier_loss + align_loss + 0.1 * loss_trg_cent
            total_loss = 0.3*classifier_loss + 0.5 * rec_loss + 0.5*adv_loss + 0.5* my_recon_loss

            # Reset gradients
            optimizer_classifier.zero_grad()
            optimizer_feature.zero_grad()

            total_loss.backward()

            optimizer_classifier.step()
            optimizer_feature.step()


            # free variables
            for d in range(num_domains):
                Cs_memory[d].detach_()
            Ct_memory.detach_()

        # set model to test
        model.train(False)

        # calculate accuracy performance
        # best_acc, features, labels = test_muda(dataset_test, model)
        best_acc, target_features, labels, Target_data, Target_label, Target_classifier = test_muda(dataset_test, model)
        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t loss: {:.4f}".format(i, best_acc, total_loss)
        args.log_file.write(log_str)
        args.log_file.flush()
        print(log_str)
        log_total_loss.append(total_loss.data)
    # save_path = f"/home/lyc/research/research_5/DA/Ablation-study/tsne/seed-sub/"/{args.target}"  
    save_path = f"/home/lyc/research/research_5/DA/Ablation-study/tsne/seed-iv-sub/{args.target}"
  
    # 确保目录存在，如果不存在则创建
    os.makedirs(save_path, exist_ok=True)

# 构建保存的.mat文件的完整路径
    mat_file_path = os.path.join(save_path, "for_tsne.mat")  # 请将 "your_filename" 替换为实际的文件名

# 将变量保存到.mat文件
    io.savemat(mat_file_path, {'save_source': inputs_source.cpu().detach().numpy(),
                           'save_target': Target_data.cpu().detach().numpy(),
                          # 'save_domain_label': save_domain_label.cpu().detach().numpy(),
                           'save_true_label': labels_source.cpu().detach().numpy(),
                           'Target_feature': target_features,
                           'Target_label': Target_label.cpu().detach().numpy(),
                        #    'Target_classifier': Target_classifier.cpu().detach().numpy(),
                           'save_source_feature': feature_outs.cpu().detach().numpy(),
                        #    'save_source_classifier': outputs_source.cpu().detach().numpy()
                           })   
    return X, ture_Y, best_acc, model, log_total_loss