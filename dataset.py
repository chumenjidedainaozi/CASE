import os
import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from torch_geometric.data.collate import collate
from data_preprocessing import CustomData


# %%
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


class CustomBatch(Batch):
    @classmethod
    def from_data_list(cls, data_list, follow_batch=None, exclude_keys=None):
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=False,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch


class DrugDataset(Dataset):
    def __init__(self, data_df, drug_graph, drug_feature):
        self.data_df = data_df
        self.drug_graph = drug_graph
        self.drug_feature = drug_feature

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        # 对比学习药物的特征
        head_feature_list = []
        tail_feature_list = []

        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_ID = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            if (Drug1_ID not in self.drug_graph.keys() or Drug2_ID not in self.drug_graph.keys() or
                    Neg_ID not in self.drug_graph.keys()):
                continue
            if (Drug1_ID not in str(self.drug_feature.keys()) or Drug2_ID not in str(self.drug_feature.keys()) or
                    Neg_ID not in str(self.drug_feature.keys())):
                continue
            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)
            n_graph = self.drug_graph.get(Neg_ID)

            pos_featrue_h = self.drug_feature[Drug1_ID]
            neg_featrue_h = self.drug_feature[Neg_ID]
            pos_featrue_t = self.drug_feature[Drug2_ID]
            neg_featrue_t = self.drug_feature[Drug2_ID]

            head_feature_list.append(pos_featrue_h)
            head_feature_list.append(neg_featrue_h)
            tail_feature_list.append(pos_featrue_t)
            tail_feature_list.append(neg_featrue_t)

            pos_pair_h = h_graph
            pos_pair_t = t_graph
            neg_pair_h = n_graph
            neg_pair_t = t_graph

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])

        rel = torch.cat(rel_list, dim=0)
        head_feature = torch.stack(head_feature_list)
        tail_feature = torch.stack(tail_feature_list)
        label = torch.cat(label_list, dim=0)

        return head_pairs, tail_pairs, rel, head_feature, tail_feature, label


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def split_train_valid(data_df, fold, val_ratio=0.2):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['Y'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df


def read_dir_feature(dir_path):
    drug_featrue = dict()
    path_features = os.listdir(dir_path)

    for path_featrue in path_features:
        path_featrue = os.path.join(dir_path, path_featrue)
        featrue = torch.load(path_featrue)
        # drug_id = path_featrue[path_featrue.rfind('/', 1) + 1:path_featrue.rfind('.pt', 1)]
        drug_id = os.path.splitext(os.path.basename(path_featrue))[0]
        drug_featrue[drug_id] = featrue
    return drug_featrue


def load_ddi_dataset(root, batch_size, fold=0):
    # 读取药物图结构
    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))
    # 读取药物特征
    drug_feature = read_dir_feature('./data/feature')

    train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_train_fold{fold}.csv'))
    test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
    train_df, val_df = split_train_valid(train_df, fold=fold)

    train_set = DrugDataset(train_df, drug_graph, drug_feature)
    val_set = DrugDataset(val_df, drug_graph, drug_feature)
    test_set = DrugDataset(test_df, drug_graph, drug_feature)
    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/twosides', batch_size=2, fold=0)

    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)
    print(node_dim, edge_dim)
    # print(data.shape)
    head_pairs, tail_pairs, rel, label = data
    print(head_pairs, tail_pairs, rel, label)
# %%
