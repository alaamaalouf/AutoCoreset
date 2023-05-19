import numpy as np
from fast_enum import FastEnum
from VectorSummarizationCoresets import *
import utils
from time import perf_counter, time
from tqdm import tqdm


class VSC(FastEnum):
    RANDOM = 0
    CARATHEODORY = 1
    MEDIAN_OF_MEANS = 2
    IMPORTANCE_SAMPLING = 3
    FW = 4


OPTIMALITY_COUNTER = 0
INITIAL_RANDOM_SOLS = 3
OPT_VAL = None
PATIENCE = 5


def patienceBasedOptimalityCriteon(M_f):
    global  OPT_VAL, OPTIMALITY_COUNTER, PATIENCE
    current_val = np.sum(M_f[:, -1])
    if OPT_VAL is None:
        OPT_VAL = np.max(np.sum(M_f, axis=0))
        OPTIMALITY_COUNTER = 0
        return True
    else:
        if OPT_VAL > current_val:
            OPT_VAL = current_val
            OPTIMALITY_COUNTER = 0
            # print('OPT_VAL updated')
        else:
            OPTIMALITY_COUNTER += 1
    #print('OPTIMALITY_COUNTER is ', OPTIMALITY_COUNTER)
    if OPTIMALITY_COUNTER < PATIENCE:
        return True
    else:
        OPT_VAL = None
        OPTIMALITY_COUNTER = 0
        return False


def generateVSCoreset(Q, m, **kwargs):
    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
        raise ValueError('WHY')
    if kwargs['VSC'] is not VSC.FW:
        idxs, weights = None, None
        for i in range(len(kwargs['size_per_class'][0])):
            if kwargs['VSC'] is VSC.RANDOM:
                temp_idxs, temp_weights = generateUniformCoreset(Q[kwargs['size_per_class'][0][i]],
                                                                 y=kwargs['labels'], w=np.ones((Q.shape[0], )),
                                                                 m=kwargs['size_per_class'][1][i],
                                                                 replace=kwargs['replace'], return_indices=True,
                                                                 maintain_classes=True)
            elif kwargs['VSC'] is VSC.MEDIAN_OF_MEANS:
                temp_idxs, temp_weights = medianOfMeans(Q[kwargs['size_per_class'][0][i]],
                                                        kwargs['size_per_class'][1][i])
            elif kwargs['VSC'] is VSC.CARATHEODORY:
                _, temp_weights, _, temp_idxs = updated_cara(Q[kwargs['size_per_class'][0][i]],
                                                             np.ones((Q[kwargs['size_per_class'][0][i]].shape[0], )),
                                                             kwargs['size_per_class'][1][i], dtype='float64')
            elif kwargs['VSC'] is VSC.IMPORTANCE_SAMPLING:
                temp_idxs, temp_weights = applyImportanceSamplingForOneMeanProblem(Q[kwargs['size_per_class'][0][i]],
                                                                                   kwargs['size_per_class'][1][i],
                                                                                   replace=kwargs['replace'])
            else:
                raise ValueError('Please add an implementation for your desired method, and update the VSC class. '
                                 'Thanks!')
            if idxs is None:
                idxs, weights = kwargs['size_per_class'][0][i][temp_idxs], temp_weights
            else:
                idxs, weights = np.hstack((idxs, kwargs['size_per_class'][0][i][temp_idxs])), \
                    np.hstack((weights, temp_weights))
        return idxs, weights
    else:
        return attainFWCoreset(Q, kwargs['chunk_idxs'], kwargs['size_per_chunk'])


def initiateAutoCore(P, y, f, m, eta, zeta, model_f, **kwargs):
    if kwargs['VSC'] is VSC.RANDOM:
        return generateUniformCoreset(P, y, m, w=kwargs['old_weights'], replace=True,
                                      maintain_classes=kwargs['maintain_classes'])
    M_f = np.zeros((P.shape[0], eta))
    C = y_C = w_C = None
    
    opt_coreset = None
    opt_cost = np.inf

    for i in range(eta):
        if kwargs['randomized_sol'] == 'uniform':
            C, y_C, w_C = generateUniformCoreset(P, y, m, w=kwargs['old_weights'], replace=kwargs['replace'],
                                                 maintain_classes=kwargs['maintain_classes'])
            sol, _ = utils.fitModel(C, y_C, w_C, model_f, np.sum(kwargs['old_weights']))
            M_f[:, i] = f(P, y, kwargs['old_weights'], sol)
            if np.any(np.isnan(f(P, y, kwargs['old_weights'], sol))) or \
                    np.any(np.isinf(f(P, y, kwargs['old_weights'], sol))):
                print('Stop there you ass')
    
    while zeta(M_f):
        if np.any(M_f.shape[1] >= kwargs['size_per_class'][1]) and kwargs['VSC'] is VSC.CARATHEODORY:
            if opt_coreset is None:
                np.save('Mf.npy', M_f)
                raise ValueError('Here we fucked up!')
            break
        C_prime, w_C = generateVSCoreset(M_f, m, **kwargs)
        C, y_C = P[C_prime], y[C_prime] 
        sol, _ = utils.fitModel(C, y_C, w_C, model_f, sum_old_weights=np.sum(kwargs['old_weights']))
        M_f = np.hstack((M_f, f(P, y, kwargs['old_weights'], sol)[:, np.newaxis]))
        if opt_cost > np.sum(M_f[:, -1]):
            opt_coreset = (C, y_C, w_C)
            opt_cost = np.sum(M_f[:, -1])
        
        if np.any(np.isnan(f(P, y, kwargs['old_weights'], sol))) or \
                np.any(np.isinf(f(P, y, kwargs['old_weights'], sol))):
            print('Stop there you ass')
    
    if opt_coreset is not None:
        C, y_C, w_C = opt_coreset[0], opt_coreset[1], opt_coreset[2]

    return C, y_C, w_C


def printResults(vals, acc, times, labels):
    print('Method \t eps \t acc \t time \t')
    for i in range(len(labels)):
        print('{} \t {} \t {} \t {} \t'.format(labels[i], vals[i], acc[i], times[i]))


def main():
    global INITIAL_RANDOM_SOLS
    file_names = ['TinyImageNet']#['CIFAR10'] #['Data/credit_card.csv', 'Data/cod-rna.txt']
    for file_name in file_names:
        print('************************** {} ***************************'.format(file_name))
        #problems_choice = ['logistic_regression', 'svm']
        problems_choice = ['logistic_regression'] #'linear_regression',
        for problem in problems_choice:
            file_name_save = 'Results/' + file_name.split('/')[-1].split('.')[0] + '_' + problem + '_dim_reduced'
        
            model_parameters = {}
            if problem == 'k_means':
                problem_type = 'clustering'
                model_parameters['k'] = 2
                model_parameters['n_init'] = 10
                model_parameters['max_iter'] = 15
            elif problem == 'linear_regression':
                problem_type = 'regression'
                model_parameters['fit_intercept'] = True
            else:
                problem_type = 'classification'
                model_parameters['fit_intercept'] = True
                model_parameters['C'] = 1
        
            X_train, y_train, X_test, y_test, is_multiclass = utils.readDataset(file_path=file_name, standarize=True,
                                                                 problem_type=problem_type)
        
            if problem not in ['k_means', 'linear_regression']:
                if is_multiclass:
                    if problem == 'svm':
                        continue
                    problem = 'multiclass_' + problem
                else:
                    problem = 'binary_' + problem
        
            loss, model_f = utils.obtainLossAndModel(problem, **model_parameters)
    
            if 'multiclass' in model_f[1]:
                INITIAL_RANDOM_SOLS = 3
    
            if 'logistic' or 'svm' in problem:
                if 'TinyImageNet' not in file_name:
                    coreset_sizes = np.linspace(50, 300, 10).astype(int)
                else:
                    coreset_sizes = np.linspace(1000, 2500, 10).astype(int)
            else:
                coreset_sizes = np.linspace(80, 300, 10).astype(int)

            trials = 16


            competing_methods = len(VSC.__dict__) - 3 # -2 to include FW
            method_labels = list(VSC.__dict__.keys())[1:competing_methods + 1]
            mean_vals = np.zeros((competing_methods, coreset_sizes.shape[0]))
            mean_acc = np.zeros((competing_methods, coreset_sizes.shape[0]))
            std_vals = np.zeros((competing_methods, coreset_sizes.shape[0]))
            std_acc = np.zeros((competing_methods, coreset_sizes.shape[0]))
            mean_times = np.zeros((competing_methods, coreset_sizes.shape[0]))
            std_times = np.zeros((competing_methods, coreset_sizes.shape[0]))
        
            zeta = lambda M: patienceBasedOptimalityCriteon(M)
            old_weights = np.ones((X_train.shape[0], ))
        
            additional_params = {'randomized_sol': 'uniform', 'replace': True,
                                 'maintain_classes': problem_type == 'classification', 'old_weights': old_weights}
            start_time_all = time()
            opt_sol, opt_model = utils.fitModel(X_train, y_train, sample_weight=None, model_f=model_f,
                                                sum_old_weights=X_train.shape[0], return_model=True)
            opt_val_cost = lambda x: np.sum(loss(X_train, y_train, np.ones((X_train.shape[0], )), x))
            if 'means' in problem:
                opt_cost = opt_model.inertia_
                opt_acc = 0
            else:
                opt_cost = opt_val_cost(opt_sol)
                opt_acc = opt_model.score(X_test, y_test if 'multiclass' not in model_f[1] else y_test.argmax(axis=1))
                print('Optimal accuracy is ', opt_acc)
                if problem_type == 'classification':
                    y_pred_all = opt_model.predict(X_test)
                    #confusion_mat = utils.confusion_matrix(y_test, y_pred_all)
                    #disp = utils.ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=opt_model.classes_)
                    #disp.plot() #cmap='bwr'
                    # plt.set_cmap('bwr')
                    #utils.plt.savefig('ConfusionMats/' + file_name_save.split('/')[1] + '_fulldata.pdf', bbox_inches='tight')
                    #utils.plt.close()
            end_time_all = time()
            opt_time = end_time_all - start_time_all
            opt_tuple = (opt_val_cost, opt_cost)
            for coreset_size_idx in range(coreset_sizes.shape[0]):
                print('********************* Sample size {} *********************'.format(coreset_sizes[coreset_size_idx]))
                if problem_type == 'classification':
                    chunks, size_per_class = \
                        utils.chunkize(np.arange(X_train.shape[0]),
                                       y_train if 'multiclass' not in model_f[1] else y_train.argmax(axis=1),
                                       coreset_sizes[coreset_size_idx], solver=problem, fair=True)
                    _, _, chunk_idxs, size_per_chunk =\
                        utils.shuffleDataViaChunks(X_train,
                                                   y_train if 'multiclass' not in model_f[1] else y_train.argmax(axis=1),
                                                   chunks)
                else:
                    chunk_idxs = [np.arange(X_train.shape[0])]
                    size_per_chunk = np.array([coreset_sizes[coreset_size_idx]])
                    size_per_class = (chunk_idxs, np.array([coreset_sizes[coreset_size_idx]]))
                additional_params['chunk_idxs'] = chunk_idxs
                additional_params['size_per_chunk'] = size_per_chunk
                additional_params['labels'] = y_train
                additional_params['size_per_class'] = size_per_class
                for i in tqdm(range(competing_methods)):
                    # print('Running now method number: ', i)
                    additional_params['VSC'] = i
                    temp_vals = np.zeros((trials, ))
                    temp_acc = np.zeros((trials, ))
                    temp_times = np.zeros((trials, ))
                    size_C = 0
                    save_confusion_mat = False
                    for trial in range(trials):
                        # print('trial number ', trial)
                        start_time_coreset = time()
                        C, y_c, v = initiateAutoCore(X_train, y_train, loss, coreset_sizes[coreset_size_idx],
                                                     INITIAL_RANDOM_SOLS, zeta, model_f, **additional_params)
                        temp_acc[trial], temp_vals[trial] = utils.evaluateCoreset(C, y_c, v, X_test, y_test, model_f,
                                                                                  np.sum(old_weights), opt_tuple,
                                                                                  None if trials != 1 else
                                                                                  file_name_save + '_' + method_labels[i] +
                                                                                  str(int(coreset_sizes[
                                                                                              coreset_size_idx])))
                        end_time_coreset = time()
                        size_C += C.shape[0] / trials
                        temp_times[trial] = end_time_coreset - start_time_coreset
                    mean_vals[i, coreset_size_idx] = np.mean(temp_vals)
                    mean_acc[i, coreset_size_idx] = np.mean(temp_acc)
                    mean_times[i, coreset_size_idx] = np.mean(temp_times)
                    std_vals[i, coreset_size_idx] = np.std(temp_vals)
                    std_acc[i, coreset_size_idx] = np.std(temp_acc)
                    std_times[i, coreset_size_idx] = np.std(temp_times)
                    #print('Coreset size is ', int(size_C), ' ,Sample size is ', coreset_sizes[coreset_size_idx]) 
                # print('Vals -- uniform, us: ', mean_vals[:, coreset_size_idx])
                # print('Acc -- uniform, us: ', mean_acc[:, coreset_size_idx])
                # print('Time -- uniform, us: ', mean_times[:, coreset_size_idx])
                printResults(mean_vals[:, coreset_size_idx], mean_acc[:, coreset_size_idx],
                             mean_times[:, coreset_size_idx], method_labels)
        
                np.savez(file_name_save, mean_vals=mean_vals, mean_times=mean_times, std_vals=std_vals, std_times=std_times, sizes=coreset_sizes, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, opt_acc=opt_acc, opt_cost=opt_cost, mean_acc=mean_acc, std_acc=std_acc, opt_time=opt_time)









if __name__ == '__main__':
    main()

