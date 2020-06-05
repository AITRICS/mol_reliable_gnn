import os
import json
# from ast import literal_eval
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--path', help="Please give a config.json file with training/model/data/param details")
args = parser.parse_args()

#PATH = './out/tox21_classification/results/GatedGCN/'
#PATH = './out/hiv_classification/results/numL8/'
#PATH = './out/zinc_regression/results/'
PATH = './out/hiv_classification/results/'
#PATH = './out/bbbp_classification/results/'
#PATH = './out/bace_classification/results/'
#PATH = './out/cd_egfr/results/'
#PATH = './out/cd_vgfr2_classification/results/'

#PATH += 'ggnn_layers_RC/

#PATH += 'result_GIN_ZINC_GPU2_14h43m49s_on_Apr_02_2020_True_mean_dtseed_12_gpu/'
PATH += args.path


filename = 'result_GCN_ZINC_GPU0_00h44m08s_on_Mar_27_2020.txt'
#print(os.listdir(PATH))
test_accs = []
test_aurocs = []
test_precisions = []
test_recalls = []
test_f1s = []
test_auprcs = []
test_eces = []

train_accs = []
train_aurocs = []
train_precisions = []
train_recalls = []
train_f1s = []
train_auprcs = []
train_eces = []

model = ''
params = ''
net_params = ''
for filename in os.listdir(PATH):
    if filename[-4:] == '.txt':
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

        test_accs.append(test_acc)
        test_aurocs.append(test_auroc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)
        test_auprcs.append(test_auprc)
        test_eces.append(test_ece)
        train_accs.append(train_acc)
        train_aurocs.append(train_auroc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        train_auprcs.append(train_auprc)
        train_eces.append(train_ece)
        
        print(test_auroc)
        f.close()

#with open(PATH+filename[:-4]+'.csv', 'w') as f:
#fr = open(PATH+filenamex[:-4]+'.csv', 'w')
fr = open(PATH[:-1] + '.csv', 'w')
fr.write(model + '\n')
fr.write(params + '\n')
fr.write(net_params + '\n')
#fr.write(','.join(test_accs) + '\n')
fr.write(','.join(test_aurocs) + '\n')
fr.write(','.join(test_precisions) + '\n')
fr.write(','.join(test_recalls) + '\n')
fr.write(','.join(test_f1s) + '\n')
fr.write(','.join(test_auprcs) + '\n')
#fr.write(','.join(train_accs) + '\n')
fr.write(','.join(train_aurocs) + '\n')
fr.write(','.join(train_precisions) + '\n')
fr.write(','.join(train_recalls) + '\n')
fr.write(','.join(train_f1s) + '\n\n')
fr.write(','.join(train_auprcs) + '\n')

fr.write('AUROC,' + str(np.mean([float(x) for x in test_aurocs])) + ',' + str(np.std([float(x) for x in test_aurocs])) + '\n')
fr.write('ECE,' + str(np.mean([float(x) for x in test_eces])) + ',' + str(np.std([float(x) for x in test_eces])) + '\n')
fr.write('Precision,' + str(np.mean([float(x) for x in test_precisions])) + ',' + str(np.std([float(x) for x in test_precisions])) + '\n')
fr.write('Recall,'+ str(np.mean([float(x) for x in test_recalls])) + ',' + str(np.std([float(x) for x in test_recalls])) + '\n')
fr.write('AUPRC,'+ str(np.mean([float(x) for x in test_auprcs])) + ',' + str(np.std([float(x) for x in test_auprcs])) + '\n')
fr.write('F1,'+ str(np.mean([float(x) for x in test_f1s])) + ',' + str(np.std([float(x) for x in test_f1s])) + '\n')
#fr.write(str(np.mean([float(x) for x in train_aurocs])) + ',' + str(np.std([float(x) for x in train_aurocs])) + '\n')
#fr.write(str(np.mean([float(x) for x in train_eces])) + ',' + str(np.std([float(x) for x in train_eces])) + '\n')
#fr.write(str(np.mean([float(x) for x in train_f1s])) + ',' + str(np.std([float(x) for x in train_f1s])) + '\n')
fr.close()

