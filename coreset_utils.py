import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit, log_expit
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from DataUtils import multilabel_train_test_split
from scipy.optimize import linprog
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm

onehot_encoder = OneHotEncoder(sparse=False)
plt.rcParams.update({'font.size': 16})


losses = {
    'binary_logistic_regression': lambda X,y,sample_weight,w, C=1:
    0.5 * sample_weight / np.sum(sample_weight) * np.linalg.norm(w[:-1]) ** 2 - C * \
    np.multiply(log_expit(np.multiply(y, X.dot(w[:-1]) + w[-1])), sample_weight),
    
    'multiclass_logistic_regression': lambda X,y,sample_weight,w: multiclassLogisticRegression(
        np.hstack((X, np.ones((X.shape[0], 1)))), y, w.T, sample_weight),
        
    'binary_svm': lambda X, y, sample_weight, w, C=1:
    0.5 * sample_weight / np.sum(sample_weight) * np.linalg.norm(w[:-1])**2 + C* np.multiply(sample_weight,
                                                    np.maximum(0, 1 - np.multiply(y, X.dot(w[:-1]) + w[-1]))),
                                                    
    'linear_regression': lambda X,y,sample_weight,w: np.multiply(sample_weight, (y - (X.dot(w[:-1]) + w[-1]))**2),
    
    'k_means': lambda X,y,sample_weight,w: np.multiply(sample_weight, np.min(cdist(X, w), axis=1)**2),
}


models = {
    'binary_logistic_regression': lambda fit_intercept, C=1: LogisticRegression(fit_intercept=fit_intercept, C=C,
                                                                                max_iter=1500),
                                                                                
    'multiclass_logistic_regression': lambda fit_intercept, C=1: LogisticRegression(fit_intercept=fit_intercept, C=C,
                                                                                    class_weight='balanced',
                                                                                    max_iter=1500, solver='sag'),
    'binary_svm': lambda C=1: svm.SVC(C=C, kernel='linear'),
    
    'linear_regression': lambda fit_intercept: LinearRegression(fit_intercept=fit_intercept),
    
    'k_means': lambda k=5,n_init=10,max_iter=15: KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init)
}



def k_means_cost(Q,centers,weights = None):
    distance_table = euclidean_distances(Q, centers)
    distance_table.sort(axis=1)
    #print(distance_table)
    distances_to_sum = distance_table[:, 0]**2

    if weights is not None:     
        cost  = np.multiply(weights,distances_to_sum)
    else: 
        cost = distances_to_sum
    
    return cost


def obtainLossAndModel(problem_name, **kwargs):
    if problem_name == 'linear_regression':
        model = models[problem_name](kwargs['fit_intercept'])
    elif 'logistic' in problem_name:
        model = models[problem_name](kwargs['fit_intercept'], kwargs['C'])
    elif 'svm' in problem_name:
        model = models[problem_name](kwargs['C'])
    elif 'means' in problem_name:
        model = models[problem_name](kwargs['k'], kwargs['n_init'], kwargs['max_iter'])
    else:
        raise ValueError('Please add desired model to models and its corresponding loss to losses. Thanks!')

    return losses[problem_name], (model, problem_name)



def multiclassLogisticRegression(X, Y, W, sample_weight):
    """
    This method is the implementation of the multiclass logistic regression los function.

    :param X: A numpy ndarray representing the input data (training data).
    :param Y: A numpy ndarray representing the one-hot encoded labels corresponding to the input data (training labels).
    :param W: A numpy ndarray containing a model for each of the classes present in Y.
    :param chunks: A list containing indices of chunks for which the loss of multiclass logistic
                   regression will be evaluated on. This is for the benefit of boosting the performance of such
                   a method. For now, such feature is depreciated and will be implemented in the future.
    :return: A loss vector describing the loss of each point with respect to the multiclass logistic regression problem.
    """

    Z = - X @ W  # a dot product between the training data and the different models
    N = X.shape[0]  # number of training data points

    # Evaluation of the log on the softmax representing the loss function of the multiclass logistic regression
    loss = 1 / N * (np.einsum('ik,ki->i', X, W[:, np.argmax(Y, axis=1)]) + logsumexp(Z, axis=1))

    return np.multiply(sample_weight, loss + np.linalg.norm(W, ord='fro') ** 2 / np.sum(sample_weight))


def readTinyImageNet(standarize=False, dim_reduce=True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torchvision.utils import make_grid
    from torchvision import models, datasets
    from torchvision import transforms as T
    
    preprocess_transform = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 
                ])
    
    preprocess_transform_pretrain = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])
    
    DATA_DIR = '' #Please insert the path to the TinyImageNet data
    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    
    def generate_dataloader(data, name, transform):
        if data is None: 
            return None
    
        # Read image files to pytorch dataset using ImageFolder, a generic data 
        # loader where images are in format root/label/filename
        # See https://pytorch.org/vision/stable/datasets.html
        if transform is None:
            dataset = datasets.ImageFolder(data, transform=T.ToTensor())
        else:
            dataset = datasets.ImageFolder(data, transform=transform)
    
        # Set options for device
        if True:
            kwargs = {"num_workers": 1}
        else:
            kwargs = {}
        
        print('****************************** ', len(dataset))
        #input()
        # Wrap image dataset (defined above) in dataloader 
        dataloader = DataLoader(dataset, batch_size=1000, 
                            shuffle=(name=="train"), 
                            **kwargs)
        
        return dataloader
    
    batch_size = 64

    trainloader = generate_dataloader(TRAIN_DIR, "train",
                                      transform=preprocess_transform)
    val_img_dir = os.path.join(VALID_DIR, 'images')
    
    
    val_data = pd.read_csv(f'{VALID_DIR}/val_annotations.txt', 
                       sep='\t', 
                       header=None, 
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])

    val_data.head()
    
    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    
    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()
    
    # Create subfolders (if not present) for validation images based on label ,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
    
    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    
    
    testloader = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform)
    X_train = []
    y_train = []
    for idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        X_train.append(torch.flatten(inputs, start_dim=1).detach().cpu().numpy())
        y_train.append(targets.detach().cpu().numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    
    print('Read entire Train')
    
    X_test = []
    y_test = []
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        X_test.append(torch.flatten(inputs, start_dim=1).detach().cpu().numpy())
        y_test.append(targets.detach().cpu().numpy())
    
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    
    print('Read entire Test')
    
    multiclass = True
    
    y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))
    
    print('there are {} classes '.format(y_test.shape[1]))
    #input()
    if dim_reduce:
        print('dim_reduce')
        DIM = int(np.sqrt(X_train.shape[0]))
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=DIM)
        svd.fit(X_train)
        print('finished fitting TruncatedSVD')
        X_train = svd.transform(X_train)
        X_test = svd.transform(X_test)
        print('finished transformation using TruncatedSVD')
    
    if standarize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    print('******************* READ DATA SUCCESSFULLY *******************')
    return X_train, y_train, X_test, y_test, multiclass


def readCIFAR10(standarize=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=1)
    
    for (inputs, targets) in trainloader:
        X_train = torch.flatten(inputs, start_dim=1).detach().cpu().numpy()
        y_train = targets.detach().cpu().numpy()
    
    for (inputs, targets) in testloader:
        X_test = torch.flatten(inputs, start_dim=1).detach().cpu().numpy()
        y_test = targets.detach().cpu().numpy()
    
    multiclass = True
    
    y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))
    
    if standarize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    print('******************* READ DATA SUCCESSFULLY *******************')
    return X_train, y_train, X_test, y_test, multiclass



def readDataset(file_path, standarize=True, train_test_split_ratio=0.3, problem_type='clustering'):
    if 'TinyImageNet' in file_path:
        return readTinyImageNet()
    elif 'CIFAR10' in file_path:
        return readCIFAR10() 
    elif "3D_spatial" in file_path or "YearPredictionMSD" in file_path:
        data = np.array(pd.read_csv(file_path, sep=",", header=None))
        labels = data[:, -1]
        Q = np.array(data[:, :-1])
    elif "HTRU" in file_path:
        data = pd.read_csv(file_path)
        data = data.values
        labels = data[:, -1]
        labels[labels == labels.min()] = -1
        labels[labels == labels.max()] = 1
        Q = data[:, :-1]
    elif "accelerometer" in file_path:
        data = pd.read_csv(file_path)
        data = data.values
        labels = data[:, -1]
        Q = data[:, :-1]
    elif 'credit_card' in file_path:
        data = pd.read_csv(file_path, encoding='latin1')
        data.info()
        data = data.drop(['ID'], axis=1)
        # data.head(10000)

        labels = np.array(data['Y'].values)
        labels = labels.flatten()
        labels[labels == 0] = -1
        Q = data.drop(['Y'], axis=1)
        Q = np.array(Q)
        # print(Q,labels)
    elif 'csv' in file_path:
        Q = np.genfromtxt(file_path, delimiter=',')  # here you put your data
        if 'SUSY' in file_path:
            labels = Q[:, 0]
            labels[labels == labels.min()] = -1
            labels[labels == labels.max()] = 1
            Q = Q[:, 1:]
        else:
            labels = Q[:, -1]
            if np.unique(labels).shape[0] <= 2:
                labels[labels == labels.min()] = -1
                labels[labels == labels.max()] = 1
            Q = Q[:, :-1]
    elif ('npz' not in file_path and 'Skin' not in file_path and 'cov' not in file_path) and ('w8a' in file_path or 'cod-rna' in file_path):
        DATA = load_svmlight_file(file_path)
        Q, labels = np.asarray(DATA[0].todense()), DATA[1]
    elif '.txt' in file_path:
        DATA = np.loadtxt(file_path)
        Q, labels = DATA[:, :-1], DATA[:, -1]
        
        if np.unique(labels).size == 2:
            labels[labels == np.min(labels)] = -1
            labels[labels == np.max(labels)] = 1
    elif 'xlsx' in file_path:
        #Q = np.genfromtxt(file_path, delimiter=',')
        data = pd.read_excel(file_path)
        data = data.values
        labels = data[:, -1]
        Q = data[:, :-1]
    else:
        try:
            DATA = np.load(file_path, allow_pickle=True)
            Q = DATA['X']
            labels = DATA['y']
        except:
            DATA = pd.read_csv(file_path)
            DATA = DATA.values
            Q = DATA[:, :-1]
            labels = DATA[:, -1]

    multiclass = np.unique(labels).size > 2

    if problem_type == 'classification':
        if not multiclass:
            X_train, X_test, y_train, y_test = train_test_split(Q, labels, test_size=train_test_split_ratio,
                                                                random_state=42)
        else:
            Y = onehot_encoder.fit_transform(labels.reshape(-1, 1))
            X_train, X_test, y_train, y_test = \
                multilabel_train_test_split(Q, Y, stratify=Y, test_size=train_test_split_ratio, random_state=42)
    else:
        X_train = X_test = Q
        y_train = y_test = labels

    if standarize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, multiclass


def fitModel(X, y, sample_weight, model_f, sum_old_weights, return_model=False, **kwargs):
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0], ))
    if model_f[1] != 'k_means':
        if model_f[1] != 'linear_regression':
            params = model_f[0].get_params()
            old_C = params['C']
            params['C'] *= float(sum_old_weights / (np.sum(sample_weight)))
            model_f[0].set_params(**params)
        model_f[0].fit(X, y if 'multiclass' not in model_f[1] else y.argmax(axis=1), sample_weight)

        if model_f[1] != 'linear_regression':
            params['C'] = old_C
            model_f[0].set_params(**params)

        return np.hstack((model_f[0].coef_.flatten(), model_f[0].intercept_)) if 'multiclass' not in model_f[1] \
            else np.hstack((model_f[0].coef_, model_f[0].intercept_[:, np.newaxis])), None if not return_model else\
            model_f[0]
    else:
        model_f[0].fit(X, y, sample_weight)
        return model_f[0].cluster_centers_, None if not return_model else model_f[0]


def evaluateCoreset(C, y_c, v, X_test, y_test, model_f, sum_old_weights, opt_val=None, file_name_confusion=None):
    sol, model = fitModel(C, y_c, sample_weight=v, model_f=model_f, sum_old_weights=sum_old_weights, return_model=True)
    if model_f[1] not in ['k_means']:
        if file_name_confusion is not None:
            y_pred = model.predict(X_test)
            confusion_mat = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=model.classes_)
            disp.plot() #cmap='bwr'
            # plt.set_cmap('bwr')
            plt.savefig('ConfusionMats/'+file_name_confusion.split('/')[1]+'.pdf', bbox_inches='tight')
            plt.close()
        return model.score(X_test, y_test if 'multiclass' not in model_f[1] else y_test.argmax(axis=1)), opt_val[0](sol) / opt_val[1] - 1
    else:
        return 0, opt_val[0](sol) / opt_val[1] - 1


def beMoreFair(sample_size, class_sizes, fair=True):
    """
    This function is responsible to distributing the sample size over different class in the training data in a fair
    manner if permitted.

    :param sample_size: The sample size chosen by the user.
    :param class_sizes: A numpy array containing the size of each class of points (classification-tasks related)
    :param fair: A boolean variable to indicate fairness among classes.
    :param first_ver: A boolean variable for choosing different fairness policy generation mechanisms.
    :return: A numpy array containing the ratio of the sample size associated with each of the traning classes.
    """

    ratios = class_sizes / np.sum(class_sizes)  # the ratio of classes
    if not fair:  # no fairness required; bigger classes dominate over the sample size
        return ratios

    # A lower bound on how much from the sample size can we take from each class
    min_bound = np.min(class_sizes / sample_size)
    if min_bound >= 1:
        min_bound = 0.8  # if the ratio is larger than 1, then the lower bound is 0

    # First policy favors bigger classes over smaller ones, the second favors smaller classes over bigger ones, and the
    # third policy aims to be "almost uniform" across all classes.
    policy_1 = ratios
    policy_2 = np.abs(np.log(ratios)) / np.sum(np.abs(np.log(ratios)))
    policy_3 = np.abs(np.exp(-ratios)) / np.sum(np.abs(np.exp(-ratios)))

    # Turn the policy vector into a matrix of 'number of classes' x 3
    W = np.vstack([policy_1, policy_2,  policy_3]).T #

    # In what follows, a linear programming is invoked.
    # Recall that linear programming problem looks like:
    # min c^T x
    # s.t.
    #      A_eq * x = b_eq
    #      A_ub * x <= b_ub
    # Specifically speaking, the idea is to introduce a weighting system to policies.

    for alpha in np.linspace(start=1, stop=.4, num=10):
        # Specifically, the weighting system aims to take a convex combination over the policies (the whole vectors).
        c = np.ones((W.shape[1],))  # the c vector
        bounds = [(min_bound * alpha, None) for _ in range(c.shape[0])]  # bounds on each entry of the weight
        A_eq = np.ones((c.shape[0], 1)).T  # A_eq is vector of ones,
        b_eq = np.ones((1,))  # b_eq is simply np.array([1])
        A_ub = sample_size * W  # A_ub * x -> is basically the number of points chosen from sample size for each class
        b_ub = class_sizes  # upper bound is the size of each of the classes

        # Solve linear programming
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        if not res.success:
            continue
        else:
            return A_ub.dot(res.x) / sample_size

    return policy_1


def chunkize(idxs, y, sample_size, K=3, fair=False, solver=None):
    """
    The following method split the data into chunks, specifically, each class into chunks where each
    chunk contains size_per_class[i] // K points. In addition, this method calls 'fairnessPolicyAdaptor' in order
    to understand how many chunks each class should contain.

    :param idxs: A numpy array of indices.
    :param y: A numpy array of labels (classification related)
    :param sample_size: A positive integer indicating the desired sample size chosen by the user.
    :param K: A positive integer which affects the number of points in each chunk.
    :param fair: A boolean variable indicating the use for fairness among classes.
    :return: An iterator of tuples containing chunks, the associated sampling size per chunk for the FW algorith,
             and finally the label corresponding with each chunk.
    """

    if solver == "linear_regression":
        unique_labels = [1]
        classes = [idxs]
    else:
        unique_labels = np.unique(y)  # unique labels

        # list of numpy array of indices, where each array corresponds to the indices of class points
        classes = [idxs[y == i] for i in unique_labels]

    # Adaptive ratios of classes, which will indicate how much points percentage of the sample size is dedicated for
    # each class
    ratios = beMoreFair(sample_size=sample_size, class_sizes=np.array([x.shape[0] for x in classes]), fair=fair)
    
   
    # Sample size per class
    size_per_class = np.round([sample_size * x for x in ratios])
    size_per_class = np.round(size_per_class / np.sum(size_per_class) * sample_size)
    #print(size_per_class, 'look');  # exit(0)
    # Splitting each of the classes into chunks
    arrays = []
    chunks_label = []
    for i, c in enumerate(classes):
        Z = np.array_split(c, size_per_class[i] // K) if size_per_class[i] > K else [c]
        chunks_label.extend([unique_labels[i] for _ in range(len(Z))])
        arrays.extend(Z)

    chunk_sizes = np.array([x.shape[0] for x in arrays])  # computing size of chunks

    # Setting the parameter indicating the sample size for FW
    epsilons = [1.0 / K for _ in range(chunk_sizes.shape[0])]

    return zip(arrays, epsilons, chunks_label), (classes, size_per_class.astype(np.int32))


def shuffleDataViaChunks(X, y, chunks):
    chunks = list(chunks)
    indices = np.hstack([x[0] for x in chunks])
    X = X[indices]
    y = y[indices]
    
    indices_chunks = np.array([[0, chunks[0][0].size] if i == 0 else
                      [np.sum([chunks[j][0].size for j in range(0,i)]),
                       np.sum([chunks[j][0].size for j in range(0,i)]) + chunks[i][0].size] for i in range(len(chunks))])
    return X, y, indices_chunks, np.array([1/chunks[j][1] for j in range(len(chunks))]).astype(int)