import numpy as np
from collections import Counter
import h5py
import os

def accuracy(out, label):
    out = np.array(out)
    label = np.array(label)
    total = out.shape[0]
    #NEW
    correct = (out == label[:total]).sum().item() / total
    return correct

def sensitivity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label == 1.)
    sens = np.sum(out[mask]) / np.sum(mask)

    return sens

def specificity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label <= 1e-5)
    total = np.sum(mask)
    spec = (total - np.sum(out[mask])) / total

    return spec

def major_vote(out):
    """

    :param out: numpy.ndarray (3, batch, 2)
    :return: (batch, 2)
    """
    out = np.argmax(out, axis=2)
    _, b = out.shape
    major = np.zeros((b, 2))
    for i in range(b):
        c = Counter(out[:,i])
        major[i, c.most_common()[0][0]] = 1
    return major

def filter_weight(w):
    filter_li = ["conv","bn","ca","sa"]
    out = {}
    for a in w.keys():
        for b in filter_li:
            if b in a:
                out[a] = w[a]

    return out

def filter_weight_module(w):
    out = {}
    for a in w.keys():
        if "module." in a:
            b = a[7:]
            out[b] = w[a]

    return out

def get_state_dict(origin_dict):
    old_keys = origin_dict.keys()
    new_dict = {}
    for ii in old_keys:
        temp_key = str(ii)
        if temp_key[0:7] == "module.":
            new_key = temp_key[7:]
        else:
            new_key = temp_key

        new_dict[new_key] = origin_dict[temp_key]
    return new_dict

class Hdf5:
    def __init__(self, dataset, is_train, data_keys, source_dir, num_source=1):
        self.dataset = dataset.lower()
        self.num_source = num_source
        self.is_train = is_train
        self.source_dir = source_dir
        self.source = self.set_source()
        self.keys = data_keys.copy()

        self.max_shape = {key:0 for key in self.keys}

    def set_source(self):
        is_train = self.is_train
        if self.dataset == "nrcki": #NEW ADDED LINE
            if is_train:
                name = "trainData_nrcki_CV{:02}.txt".format(self.num_source)
            else:
                name = "testData_nrcki.txt"
        else:
            raise ValueError("invaild dataset")

        return os.path.join(self.source_dir,name)

    def get_patchlist(self):
        return self.source

    def getDataDicByIndex(self, index):
        with open(self.source, 'r') as f:
            files = f.readlines()
        data = {}
        file = files[index].strip()
        with h5py.File(file, "r") as hf:
            if "fdata" not in hf.keys() and "fdata" in self.keys:
                self.keys.remove("fdata")
            elif "fdata_new" not in hf.keys() and "fdata_new" in self.keys:
                self.keys.remove("fdata_new")
            for k in self.keys:
                if k not in hf.keys():
                    raise Exception("\"{}\" is Invaild key, choose one of {}".format(k, tuple(hf.keys())))
                data[k] = np.array(hf[k])
        return data

    def getDataDicByName(self, name):
        data = {}
        with h5py.File(name, "r") as hf:
            if "fdata" not in hf.keys() and "fdata" in self.keys:
                self.keys.remove("fdata")
            if "fdata_new" not in hf.keys() and "fdata_new" in self.keys: #NEW avant c'était ELIF -> we don't have fdata
                self.keys.remove("fdata_new")
            for k in self.keys:
                if k not in hf.keys():
                    print(name, k)
                    raise Exception("\"{}\" is Invaild key, choose one of {}".format(k, tuple(hf.keys())))
                data[k] = np.array(hf[k])
                #data[k] = gaussian_noise(data[k])
        return data

    def getBatchDicByNames(self, names):
        ndim = {}
        for i, name in enumerate(names):
            if i == 0:
                batch = self.getDataDicByName(name)
                for k in self.keys:
                    ndim[k] = batch[k].ndim
                    if k != 'label': #NEW
                        if self.max_shape[k] >= batch[k].shape[-1]:
                            batch[k] = np.pad(batch[k],[[0,0]]*(ndim[k] - 1)+[[0,self.max_shape[k]-batch[k].shape[-1]]])
                        else:
                            self.max_shape[k] = batch[k].shape[-1]
                    #NEW
                    else:
                        batch[k] = np.array(batch[k])

            else:
                data = self.getDataDicByName(name)
                for k in self.keys:
                    if k != 'label':#NEW
                        if self.max_shape[k] >= data[k].shape[-1]:
                            data[k] = np.pad(data[k], [[0,0]]*(ndim[k] - 1)+[[0,self.max_shape[k]-data[k].shape[-1]]])
                        else:
                            self.max_shape[k] = data[k].shape[-1]
                            batch[k] = np.pad(batch[k],[[0,0]]*(ndim[k] - 1)+[[0,self.max_shape[k]-batch[k].shape[-1]]])
                        batch[k] = np.concatenate([batch[k], data[k]], axis=0)
                    #NEW
                    else:
                        #print(data['label'], batch['label'])
                        batch[k] = np.append(batch[k], np.array(data[k]))

        return batch

    def infoGain(self):
        with open(self.source, 'r') as f:
            files = f.readlines()

        total = len(files)
        count = 0
        files = files
        for file in files:
            with h5py.File(file.strip(), "r") as hf:
                if hf["label"][()] == 1: #hf["label"][0][0]
                    count += 1
        return count / total, 1 - count / total

    def count(self):
        with open(self.source, 'r') as f:
            files = f.readlines()

        total = len(files)
        count = 0
        files = files
        for file in files:
            with h5py.File(file.strip(), "r") as hf:
                if hf["label"][0][0] == 1.:
                    count += 1
        return count, total


def get_sample_from_dir(dir):
    dir = dir.split("\\")[-2]
    return int(dir[6:])
