class defaultConfig(object):
    # hyper_para
    epoch_num = 200
    batch_size = 64
    learning_rate = 0.001
    input_size = 166  # lock
    hidden_size = 256  # lock 256
    num_layers = 3  # lock 3
    output_size = 1  # lock
    seq_length = 10  # lock
    dropout = 0.4  # 0

    # rige
    batch_size_r = [512, 128, 256, 64, 1024]
    learning_rate_r = [0.001, 0.0005, 0.01]
    hidden_size_r = [256, 512]  # lock 256
    num_layers_r = [2, 3, 4]
    dropout_r = [0.1, 0.2, 0.3, 0.4]

    # next three are mutex
    train = 1
    test1 = 0  # mseloss
    test2 = 0  # R2

    loadBool = 0  # load module from module_path if value equal to 1, if not create new module accord to hyper-para
    module_path = "./module/NetV3.0-470.pth"

    # trainPath
    features_path1 = "./trainset/features1_nor.npy"  # train_feature
    targets_path1 = "./trainset/targets1.npy"  # train_targets

    features_path2 = "./trainset/features2_nor.npy"
    targets_path2 = "./trainset/targets2.npy"

    features_path3 = "./trainset/features3_nor.npy"
    targets_path3 = "./trainset/targets3.npy"

    features_path4 = "./trainset/features4_nor.npy"
    targets_path4 = "./trainset/targets4.npy"

    features_path5 = "./trainset/features5_nor.npy"
    targets_path5 = "./trainset/targets5.npy"

    features_path6 = "./trainset/features6_nor.npy"
    targets_path6 = "./trainset/targets6.npy"

    features_path7 = "./trainset/features7_nor.npy"
    targets_path7 = "./trainset/targets7.npy"

    features_path8 = "./trainset/features8_nor.npy"
    targets_path8 = "./trainset/targets8.npy"

    # testPath
    features_path = "./trainset/features8_nor.npy" # test_feature
    targets_path = "./trainset/targets8.npy"  # test_targets
    # features_path = "./testset/2020_features1_nor.npy"
    # targets_path = "./testset/2020_targets1.npy" 