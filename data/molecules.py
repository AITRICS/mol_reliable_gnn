import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from rdkit import Chem

from data.splitters import random_scaffold_split, generate_scaffold

# *NOTE
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']

ATOM_VOCAB = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
              'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
              'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
              'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))

def edge_feature(bond):
    bt = bond.GetBondType()
    return np.asarray([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()])

def atom_feature(atom, use_atom_meta):
    #return np.asarray(
    #    one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB))
    if use_atom_meta == False:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) 
            )
    else:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) +
            one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
            [atom.GetIsAromatic()])


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, smiles_list, label_list, name, use_atom_meta=True):
        self.smiles_list = smiles_list
        self.label_list = label_list
        self.num_graphs = len(smiles_list)
        self.name = name
        self.use_atom_meta = use_atom_meta
        
        self.graph_lists = []
        self.graph_labels = []
        self.graph_smiles = []
        self._prepare()
        self.n_samples = len(self.graph_lists)

    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.name.upper()))
        
        for idx, smiles in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smiles)


            # To generate node features
            atom_list = mol.GetAtoms()
            num_atoms = len(atom_list)
            node_features = np.asarray([atom_feature(atom, self.use_atom_meta) for atom in atom_list])

            # To generate adj matrix and edge features
            bond_list = mol.GetBonds()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(num_atoms)
            g.ndata['feat'] = node_features
            
            for bond in bond_list:
                srt = bond.GetBeginAtom().GetIdx()
                dst = bond.GetEndAtom().GetIdx()
                g.add_edges(srt, dst)

            edge_features = np.asarray([edge_feature(bond) for bond in bond_list])

            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            if type(self.label_list[idx]) == np.ndarray:
                #   for chembl pretrain case
                self.graph_labels.append(self.label_list[idx].astype('float32'))
            elif type(self.label_list[idx]) == list:
                self.graph_labels.append([float(x) for x in self.label_list[idx]])
            else:
                self.graph_labels.append(float(self.label_list[idx]))

            self.graph_smiles.append(smiles)
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx], self.graph_smiles[idx]
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, smiles_list, label_list, params, num_train=50000, num_val=5000, num_test=5000):
        t0 = time.time()

        num_train_slice = -1

        # set num classes for classification task

        if params['task'] == 'classification':
            if params['dataset'] == 'CHEMBL_PRE':
                self.num_classes = 1310
            else:
                self.num_classes = 1
        
        # using multiple meta-info: 'atom_type, degree, numHs, implicit_valence, aromaticity'
        # bond features: 'SINGLE, DOUBLE, TRIPLE, AROMATIC'
        if params['atom_meta'] == False: 
            self.num_atom_type = 40
        else:
            self.num_atom_type = 58
            
        self.num_bond_type = 6

        # Different splitting strategies for regression & classification

        if params['task'] == 'regression':
            smiles_train = smiles_list[:num_train]
            if num_train_slice != -1:
                smiles_train = smiles_list[:num_train_slice]
            smiles_val = smiles_list[num_train:num_train+num_val]
            smiles_test = smiles_list[num_train+num_val:]

            label_train = label_list[:num_train]
            if num_train_slice != -1:
                smiles_train = smiles_list[:num_train_slice]
            label_val = label_list[num_train:num_train+num_val]
            label_test = label_list[num_train+num_val:]
        else:
            if params['scaffold_split'] == True:
                frac = [0.8, 0.1, 0.1]

                smiles_train, smiles_val, smiles_test, label_train, label_val, label_test = \
                    random_scaffold_split(label_list, smiles_list, frac_train=frac[0], frac_valid=frac[1], frac_test=frac[2], seed=params['data_seed'])
                #   Test scaffold split complete
                scaffolds_train = set([generate_scaffold(x) for x in smiles_train])
                scaffolds_val = set([generate_scaffold(x) for x in smiles_val])
                scaffolds_test = set([generate_scaffold(x) for x in smiles_test])
                print('train/valid scaffold split: ', 
                    scaffolds_train.isdisjoint(scaffolds_val))
                print('train/test scaffold split: ', 
                    scaffolds_train.isdisjoint(scaffolds_test))
                print('valid/test scaffold split: ', 
                    scaffolds_val.isdisjoint(scaffolds_test))
            else:
                smiles_train = smiles_list[:num_train]
                smiles_val = smiles_list[num_train:num_train+num_val]
                smiles_test = smiles_list[num_train+num_val:]

                label_train = label_list[:num_train]
                label_val = label_list[num_train:num_train+num_val]
                label_test = label_list[num_train+num_val:]

        
        self.train = MoleculeDGL(smiles_train, label_train, 'TRAIN', params['atom_meta'])
        self.val = MoleculeDGL(smiles_val, label_val, 'VALIDATION', params['atom_meta'])
        self.test = MoleculeDGL(smiles_test, label_test, 'TEST', params['atom_meta'])

        # Smiles list for output verification
        self.smiles_val = smiles_val
        self.smiles_test = smiles_test

        print("Time taken: {:.4f}s".format(time.time()-t0))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels, smiles = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]

        # zero-edged molecule error occurred
        tab_snorm_e = []
        for size in tab_sizes_e:
            if size > 0:
                tab_snorm_e.append(torch.FloatTensor(size,1).fill_(1./float(size)))
            else:
                tab_snorm_e.append(torch.FloatTensor(1, 1).fill_(1./float(1)))
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e, smiles
    
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

