import os
import numpy as np
import pickle
import argparse
from train.metrics import binary_class_perfs

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="Please give a config.json file with training/model/data/param details")
args = parser.parse_args()

PATH = './out/tox21_classification/outputs/'
#PATH += 'ensemble_ggnn_swa_ver1/'
PATH += args.path

seeds = ['41', '95', '12', '35', '42', '96', '13', '36']
TOX21_CLASSES = ["nr_ar", "nr_ar_lbd", "nr_ahr", "nr_aromatase", "nr_er_lbd", "nr_er", 
            "nr_ppar_gamma", "sr_are", "sr_atad5", "sr_hse", "sr_mmp", "sr_p53"]

def np_sigmoid(x):
    return (1./(1. + np.exp(-x)))

def get_ensemble_output(path, output_list):
    '''
    input: list of pickle files, under same data seed
    '''
    smiles_target_list = {}
    smiles_score_list = {}
    for output in output_list:
        with open(path + output, 'rb') as f:
            out = pickle.load(f)
            smiles = out['test_smiles']
            # scores = [np_sigmoid(x) for x in out['test_scores']]
            scores = out['test_scores']
            targets = out['test_targets']
            
        for i, x in enumerate(targets):
            '''
            # output smiles need to be configured, currently not matched with labels in the output list

            if smiles[i] not in smiles_target_list.keys():
                smiles_target_list[smiles[i]] = x
                smiles_score_list[smiles[i]] = [scores[i]]
            else:
                if x != smiles_target_list[smiles[i]]:
                    print(smiles[i])
                    print(x)
                    print(smiles_target_list[smiles[i]])
                    raise AssertionError()

                smiles_score_list[smiles[i]].append(scores[i])
            '''
            smiles_target_list[i] = x
            if i not in smiles_score_list.keys():
                smiles_score_list[i] = [scores[i]]
            else:
                smiles_score_list[i].append(scores[i])
    return smiles_score_list, smiles_target_list
    
dataseed = {}
ensemble_seed_perfs = {}
single_seed_perfs = {}
for seed in seeds:

    seed_output = {}
    for x in os.listdir(PATH):
        iddx = x.find('dtseed')+7
        if x[iddx:iddx+2] == seed:
            for tox_type in TOX21_CLASSES:
                idddx = x.find('TOX21')
                iddddx = x.find('_GPU')
                if x[idddx+6:iddddx] == tox_type:
                    try:
                        seed_output[tox_type].append(x)
                    except:
                        seed_output[tox_type] = [x]

    ensemble_seed_perfs[seed] = {}
    single_seed_perfs[seed] = {}
    for key, value in seed_output.items():

        
        smiles_score_list, smiles_target_list = get_ensemble_output(PATH, value)
        ensemble_average = np.array([np.mean(x) for x in smiles_score_list.values()])
        single_preds = np.array([x[0] for x in smiles_score_list.values()])
        
        ensemble_seed_perfs[seed][key] = binary_class_perfs(ensemble_average, np.array([x for x in smiles_target_list.values()]))
        single_seed_perfs[seed][key] = binary_class_perfs(single_preds, np.array([x for x in smiles_target_list.values()]))

#print(ensemble_seed_perfs['42'])
#print(single_seed_perfs['42'])
    
#   write file

fr = open(PATH + PATH.split('/')[-2] + '.csv', 'w')
for tox_type in TOX21_CLASSES:
    fr.write(','.join(["Tox type : ", tox_type]) + '\n')
    fr.write('test auroc,' + ','.join([str(ensemble_seed_perfs[seed][tox_type]['auroc']) for seed in seeds])+ '\n')
    fr.write('test accuracy,' + ','.join([str(ensemble_seed_perfs[seed][tox_type]['accuracy']) for seed in seeds])+ '\n')
    fr.write('test precision,' + ','.join([str(ensemble_seed_perfs[seed][tox_type]['precision']) for seed in seeds])+ '\n')
    fr.write('test recall,' + ','.join([str(ensemble_seed_perfs[seed][tox_type]['recall']) for seed in seeds])+ '\n')
    fr.write('test f1,' + ','.join([str(ensemble_seed_perfs[seed][tox_type]['f1']) for seed in seeds])+ '\n')
    fr.write('test aupr,' + ','.join([str(ensemble_seed_perfs[seed][tox_type]['auprc']) for seed in seeds])+ '\n')
    fr.write('test ece,' + ','.join([str(ensemble_seed_perfs[seed][tox_type]['ece']) for seed in seeds])+ '\n')

    fr.write('\n')

fr.write(',total,,,' + ',,,'.join(TOX21_CLASSES)+',,,\n')
fr.write('test_auroc,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([ensemble_seed_perfs[seed][tox_type]['auroc'] for seed in seeds])) + ',')
    fr.write(str(np.std([ensemble_seed_perfs[seed][tox_type]['auroc'] for seed in seeds])) + ',,')
fr.write('\n')

fr.write('test_ece,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([ensemble_seed_perfs[seed][tox_type]['ece'] for seed in seeds])) + ',')
    fr.write(str(np.std([ensemble_seed_perfs[seed][tox_type]['ece'] for seed in seeds])) + ',,')
fr.write('\n')

fr.write('test_precision,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([ensemble_seed_perfs[seed][tox_type]['precision'] for seed in seeds])) + ',')
    fr.write(str(np.std([ensemble_seed_perfs[seed][tox_type]['precision'] for seed in seeds])) + ',,')
fr.write('\n')

fr.write('test_recall,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([ensemble_seed_perfs[seed][tox_type]['recall'] for seed in seeds])) + ',')
    fr.write(str(np.std([ensemble_seed_perfs[seed][tox_type]['recall'] for seed in seeds])) + ',,')
fr.write('\n')

fr.write('test_auprc,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([ensemble_seed_perfs[seed][tox_type]['auprc'] for seed in seeds])) + ',')
    fr.write(str(np.std([ensemble_seed_perfs[seed][tox_type]['auprc'] for seed in seeds])) + ',,')
fr.write('\n')

fr.write('test_f1,,,,')
for i, tox_type in enumerate(TOX21_CLASSES):
    fr.write(str(np.mean([ensemble_seed_perfs[seed][tox_type]['f1'] for seed in seeds])) + ',')
    fr.write(str(np.std([ensemble_seed_perfs[seed][tox_type]['f1'] for seed in seeds])) + ',,')
fr.write('\n')

