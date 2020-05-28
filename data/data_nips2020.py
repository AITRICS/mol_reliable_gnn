"""
    File to load dataset based on user control from main file
"""
import csv
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, CalcTPSA

# from data.molecules_nips2020 import MoleculeDataset
from data.molecules_nips2020 import MoleculeDatasetDGL

from data.preprocess_pretrain import _load_chembl_with_labels_dataset 

def read_smi_and_label(DATASET_NAME, num_train, num_val, num_test, seed, shuffle=True):
    print('Seed Number of Data: '+str(seed))
    rand_state = np.random.RandomState(int(seed))
    with open('./data/nips2020/' + DATASET_NAME + '.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        contents = np.asarray([
            (row['smiles'], float(row['label'])) for row in reader if row['label'] != ''
        ])

    num_total = num_train + num_val + num_test
    contents = contents[:num_total]
    if shuffle:
        rand_state.shuffle(contents)

    return contents[:,0], contents[:,1]


def read_smi_and_label_pretrain(DATASET_NAME, num_train, num_classes, seed, shuffle=True):
    print('Seed Number of Data: '+str(seed))
    rand_state = np.random.RandomState(int(seed))
    with open('./data/nips2020/' + DATASET_NAME + '.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        contents = np.asarray([
            (row['smiles'], [row['label'+str(i+1)] for i in range(num_classes)]) for row in reader 
        ])

    if shuffle:
        rand_state.shuffle(contents)

    return contents[:,0], contents[:,1]

def get_Chembl_pretrain_dataset(seed, shuffle=True):
    print('Seed Number of Data: '+str(seed))
    rand_state = np.random.RandomState(int(seed))
    '''
    root_path = 'data/dataset/chembl_filtered/raw'
    smiles_list, rdkit_mol_objs, folds, labels = _load_chembl_with_labels_dataset (root_path)
    num_total = 430000

    #   temporally set to sample data
    with open('./data/dataset/chembl_filtered/smiles_list_sample.pkl', 'rb') as f:
        smiles_list = pickle.load(f)
    with open('./data/dataset/chembl_filtered/labels_sample.pkl', 'rb') as f:
        labels = pickle.load(f)
    num_total = 100
    contents = np.asarray([
        (smiles_list[i], labels[i]) for i in range(num_total)
    ])
    '''

    #   temporally set to sample data
    # with open('./data/nips2020/' + DATASET_NAME + '.csv', newline='') as csvfile:
    # DATASET_NAME = 'chembl_sample'
    DATASET_NAME = 'chembl'
    NUM_CLASSES = 1310
    with open('./data/nips2020/' + DATASET_NAME + '.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        contents = np.asarray([
            # (row[0], [float(x) for x in row[1:]]) for row in reader 
            (row['smiles'], [row['label'+str(i+1)] for i in range(NUM_CLASSES)]) for row in reader 
        ])

    if shuffle:
        rand_state.shuffle(contents)

    return contents[:,0], contents[:,1]

def get_ZINC_pretrain_dataset(seed, shuffle=True):
    smiles_list, _ = read_smi_and_label('ZINC', 250000, 0, 0, seed)
    for i, x in enumerate(smiles_list):
        m = Chem.MolFromSmiles(x) 
        '''
        need to define which rdkit module to be used to get labels
        '''


def read_smi_and_label_tox21(seed, shuffle=True):

    print('Seed Number of Data: '+str(seed))
    rand_state = np.random.RandomState(int(seed))

    dataset_name = 'tox21_mnet'
    num_classes = 12
    TOX21_CLASSES = ["nr_ar", "nr_ar_lbd", "nr_ahr", "nr_aromatase", "nr_er_lbd", "nr_er", 
           "nr_ppar_gamma", "sr_are", "sr_atad5", "sr_hse", "sr_mmp", "sr_p53"]

    contents = {}
    with open('./data/nips2020/' + dataset_name + '.csv', newline='') as csvfile:
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
                    
def load_pretrain_data(DATASET_NAME, num_classes, seed, params):
    smiles_list, label_list = read_smi_and_label_pretrain(DATASET_NAME, num_train, num_classes, seed)


def load_data(DATASET_NAME, num_train, num_val, num_test, seed, params):

    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """

    smiles_list, label_list = read_smi_and_label(DATASET_NAME, num_train, num_val, num_test, seed)

    return MoleculeDatasetDGL(smiles_list, label_list, params, num_train, num_val, num_test)


    

# def load_data_tox21(DATASET_NAME, num_train, num_val, num_test, seed, params):
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


    #return MoleculeDatasetDGL(smiles_list, label_list, params, num_train, num_val, num_test)
    return data_objs


if __name__ == "__main__" :
   read_smi_and_label_tox2dude('nr_ar', 44)
