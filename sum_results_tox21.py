import os
import json
from ast import literal_eval
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--path', help="Please give a config.json file with training/model/data/param details")
args = parser.parse_args()

PATH = './out/tox21_classification/results/'
#PATH = './out/hiv_classification/results/numL8/'
#PATH = './out/zinc_regression/results/'
#PATH = './out/hiv_classification/results/'
#PATH = './out/bbbp_classification/results/'
#PATH = './out/bace_classification/results/'
#PATH += 'ggnn_layers_RC/

#PATH += 'result_GIN_ZINC_GPU2_14h43m49s_on_Apr_02_2020_True_mean_dtseed_12_gpu/'
PATH += args.path


# filename = 'result_GCN_ZINC_GPU0_00h44m08s_on_Mar_27_2020.txt'
# print(os.listdir(PATH))
TOX21_CLASSES = ["nr_ar", "nr_ar_lbd", "nr_ahr", "nr_aromatase", "nr_er_lbd", "nr_er", 
            "nr_ppar_gamma", "sr_are", "sr_atad5", "sr_hse", "sr_mmp", "sr_p53"]

#TOX21_CLASSES = ["nr_ar", "nr_er", "nr_ppar_gamma", "sr_hse"]

test_accs = {}
test_aurocs = {}
test_precisions = {}
test_recalls = {}
test_f1s = {}
test_auprcs = {}
test_eces = {}

train_accs = {}
train_aurocs = {}
train_precisions = {}
train_recalls = {}
train_f1s = {}
train_auprcs = {}
train_eces = {}

for tox_type in TOX21_CLASSES:
    test_accs[tox_type] = []
    test_aurocs[tox_type] = []
    test_precisions[tox_type] = []
    test_recalls[tox_type] = []
    test_f1s[tox_type] = []
    test_auprcs[tox_type] = []
    test_eces[tox_type] = []

    train_accs[tox_type] = []
    train_aurocs[tox_type] = []
    train_precisions[tox_type] = []
    train_recalls[tox_type] = []
    train_f1s[tox_type] = []
    train_auprcs[tox_type] = []
    train_eces[tox_type] = []

    for filename in os.listdir(PATH):
        iddx = filename.find('TOX21')
        #iddx = filename.find('HIV')
        idddx = filename.find('GPU')

        if (filename[-4:] == '.txt') & (tox_type == filename[iddx+6:idddx-1]):
            filenamex = filename
            #with open(PATH+filename, 'r') as f:
            f = open(PATH+filename, 'r')
            res = f.readlines()
            res = [x.replace('\n', '') for x in res]
            res = [x for x in res if x]

            model = res[1]
            params = res[2]
            net_params = res[3]

            test_acc = res[-16][-6:]
            test_auroc = res[-15][-6:]
            test_precision = res[-14][-6:]
            test_recall = res[-13][-6:]
            test_f1 = res[-12][-6:]
            test_auprc = res[-11][-6:]
            test_ece = res[-10][-6:]

            train_acc = res[-9][-6:]
            train_auroc = res[-8][-6:]
            train_precision = res[-7][-6:]
            train_recall = res[-6][-6:]
            train_f1 = res[-5][-6:]
            train_auprc = res[-4][-6:]
            train_ece = res[-3][-6:]

            test_accs[tox_type].append(test_acc)
            test_aurocs[tox_type].append(test_auroc)
            test_precisions[tox_type].append(test_precision)
            test_recalls[tox_type].append(test_recall)
            test_f1s[tox_type].append(test_f1)
            test_auprcs[tox_type].append(test_auprc)
            test_eces[tox_type].append(test_ece)
            train_accs[tox_type].append(train_acc)
            train_aurocs[tox_type].append(train_auroc)
            train_precisions[tox_type].append(train_precision)
            train_recalls[tox_type].append(train_recall)
            train_f1s[tox_type].append(train_f1)
            train_auprcs[tox_type].append(train_auprc)
            train_eces[tox_type].append(train_ece)
            '''
            test_acc = res[-14][-6:]
            test_auroc = res[-13][-6:]
            test_precision = res[-12][-6:]
            test_recall = res[-11][-6:]
            test_f1 = res[-10][-6:]
            test_auprc = res[-9][-6:]

            train_acc = res[-8][-6:]
            train_auroc = res[-7][-6:]
            train_precision = res[-6][-6:]
            train_recall = res[-5][-6:]
            train_f1 = res[-4][-6:]
            train_auprc = res[-3][-6:]

            test_accs[tox_type].append(test_acc)
            test_aurocs[tox_type].append(test_auroc)
            test_precisions[tox_type].append(test_precision)
            test_recalls[tox_type].append(test_recall)
            test_f1s[tox_type].append(test_f1)
            test_auprcs[tox_type].append(test_auprc)
            # test_eces[tox_type].append(test_ece)
            train_accs[tox_type].append(train_acc)
            train_aurocs[tox_type].append(train_auroc)
            train_precisions[tox_type].append(train_precision)
            train_recalls[tox_type].append(train_recall)
            train_f1s[tox_type].append(train_f1)
            train_auprcs[tox_type].append(train_auprc)
            # train_eces[tox_type].append(train_ece)
            '''
            
            print(test_auroc)
            f.close()

    #with open(PATH+filename[:-4]+'.csv', 'w') as f:
#fr = open(PATH+filenamex[:-4]+'.csv', 'w')
fr = open(PATH[:-1] + '.csv', 'w')
fr.write(model + '\n')
fr.write(params + '\n')
fr.write(net_params + '\n')

for tox_type in TOX21_CLASSES:
    fr.write(','.join(["Tox type : ", tox_type]) + '\n')
    #fr.write(','.join(test_accs) + '\n')
    fr.write('test auroc,' + ','.join(test_aurocs[tox_type]) + '\n')
    fr.write('test accuracy,' + ','.join(test_accs[tox_type]) + '\n')
    fr.write('test precision,' + ','.join(test_precisions[tox_type]) + '\n')
    fr.write('test recall,' + ','.join(test_recalls[tox_type]) + '\n')
    fr.write('test f1,' + ','.join(test_f1s[tox_type]) + '\n')
    fr.write('test aupr,' + ','.join(test_auprcs[tox_type]) + '\n')
    fr.write('test ece,' + ','.join(test_eces[tox_type]) + '\n')

    #fr.write(','.join(train_accs) + '\n')
    fr.write('train auroc,' + ','.join(train_aurocs[tox_type]) + '\n')
    fr.write('train accu,' + ','.join(train_accs[tox_type]) + '\n')
    fr.write('train precision,' + ','.join(train_precisions[tox_type]) + '\n')
    fr.write('train recall,' + ','.join(train_recalls[tox_type]) + '\n')
    fr.write('train f1,' + ','.join(train_f1s[tox_type]) + '\n')
    fr.write('train auprc,' + ','.join(train_auprcs[tox_type]) + '\n')
    fr.write('train ece,' + ','.join(train_eces[tox_type]) + '\n')

fr.write('\n')
auroc_avgs = {}
auroc_stds = {}
precision_avgs = {}
precision_stds = {}
recall_avgs = {}
recall_stds = {}
f1_avgs = {}
f1_stds = {}


fr.write(',total,,,' + ',,,'.join(TOX21_CLASSES)+',,,\n')
fr.write('test_auroc,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([float(x) for x in test_aurocs[tox_type]])) + ',')
    fr.write(str(np.std([float(x) for x in test_aurocs[tox_type]])) + ',,')
fr.write('\n')

fr.write('test_ece,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([float(x) for x in test_eces[tox_type]])) + ',')
    fr.write(str(np.std([float(x) for x in test_eces[tox_type]])) + ',,')
fr.write('\n')

'''
fr.write('test_accuracy,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([float(x) for x in test_accs[tox_type]])) + ',')
    fr.write(str(np.std([float(x) for x in test_accs[tox_type]])) + ',,')
fr.write('\n')
'''

fr.write('test_precision,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([float(x) for x in test_precisions[tox_type]])) + ',')
    fr.write(str(np.std([float(x) for x in test_precisions[tox_type]])) + ',,')
fr.write('\n')
fr.write('test_recall,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([float(x) for x in test_recalls[tox_type]])) + ',')
    fr.write(str(np.std([float(x) for x in test_recalls[tox_type]])) + ',,')
fr.write('\n')

fr.write('test_auprc,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([float(x) for x in test_auprcs[tox_type]])) + ',')
    fr.write(str(np.std([float(x) for x in test_auprcs[tox_type]])) + ',,')
fr.write('\n')

fr.write('test_f1,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([float(x) for x in test_f1s[tox_type]])) + ',')
    fr.write(str(np.std([float(x) for x in test_f1s[tox_type]])) + ',,')
fr.write('\n')

fr.close()

