"""
    File to load dataset based on user control from main file
"""
import csv
import numpy as np
import pickle

'''
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, CalcTPSA
'''

from data.molecules import MoleculeDatasetDGL

def read_smi_and_label(DATASET_NAME, num_train, num_val, num_test, seed, shuffle=True):
    print('Seed Number of Data: '+str(seed))
    rand_state = np.random.RandomState(int(seed))
    with open('./data/datasets/' + DATASET_NAME + '.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        contents = np.asarray([
            (row['smiles'], float(row['label'])) for row in reader if row['label'] != ''
        ])

    num_total = num_train + num_val + num_test
    contents = contents[:num_total]
    if shuffle:
        rand_state.shuffle(contents)

    return contents[:,0], contents[:,1]

def read_smi_and_label_tox21(seed, shuffle=True):

    print('Seed Number of Data: '+str(seed))
    rand_state = np.random.RandomState(int(seed))

    dataset_name = 'tox21_mnet'
    num_classes = 12
    TOX21_CLASSES = ["nr_ar", "nr_ar_lbd", "nr_ahr", "nr_aromatase", "nr_er_lbd", "nr_er", 
           "nr_ppar_gamma", "sr_are", "sr_atad5", "sr_hse", "sr_mmp", "sr_p53"]

    contents = {}
    with open('./data/datasets/' + dataset_name + '.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        contents = {}
        for class_n in TOX21_CLASSES:
            contents[class_n] = []
        for row in reader:
            for class_n in TOX21_CLASSES:
                if row[class_n]:
                    contents[class_n].append((row['smiles'], float(row[class_n])))

        for class_n in TOX21_CLASSES:
            contents[class_n] = np.asarray(contents[class_n])
            if shuffle:
                rand_state.shuffle(contents[class_n])

    return contents


def load_data(DATASET_NAME, num_train, num_val, num_test, seed, params):

    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """

    smiles_list, label_list = read_smi_and_label(DATASET_NAME, num_train, num_val, num_test, seed)

    return MoleculeDatasetDGL(smiles_list, label_list, params, num_train, num_val, num_test)


def load_data_tox21(DATASET_NAME, seed, params):

    """
        This function is called in the main.py file 
        returns:
        ; a list of dataset object
    """
    data_objs = []

    contents = read_smi_and_label_tox21(seed)

    for key, value in contents.items():
        num_total = len(value)
        num_train = num_total // 10 * 8
        num_val = num_total // 10
        num_test = num_total // 10
        data_obj = MoleculeDatasetDGL(value[:, 0], value[:, 1], params,
                         num_train, num_val, num_test)
        data_obj.tox_type = key
        data_obj.num_classes = 1
        data_objs.append(data_obj)

    return data_objs

