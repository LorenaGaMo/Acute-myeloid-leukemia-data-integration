import numpy as np
import pandas as pd
import os


def to_categorical(data, dtype=None):
    val_to_cat = {}
    cat = []
    index = 0
    for val in data:
        if dtype == 'ic':
            if val not in ['1', '2', '3', '4ER+', '4ER-', '5', '6', '7', '8', '9', '10']:
                val = '1'
            if val in ['4ER+','4ER-']:
                val='4'
        if val not in val_to_cat:
            val_to_cat[val] = index
            cat.append(index)
            index += 1
        else:
            cat.append(val_to_cat[val])
    return np.array(cat)



def get_data(data):

    d = {}
    clin_fold = data[["Sample"]]

    exp = data[[col for col in data if col.startswith('gene')]]
    meth = data[[col for col in data if col.startswith('meth')]]
    mirna = data[[col for col in data if col.startswith('mirna')]]
    d['vs'] = list(data['clin:vital_status'].values)

    d['expnp'] = exp.astype(np.float32).values
    d['methnp'] = meth.astype(np.float32).values 
    d['mirnanp'] = mirna.astype(np.float32).values 
    d['vsnp'] = to_categorical(d['vs'])


    
    # We have to get the entire dataset, transform them into one-hots, bins
    complete_data = r"../data/omicsDataOS.csv" #Ivan tiene omicsData_3.csv
    # complete_data = pd.read_csv(complete_data).set_index("METABRIC_ID")
    complete_data =  pd.read_csv(complete_data, index_col=None, header=0)

    
    return d

def normalizeRNA(*args):
    if len(args) > 1: 
        normalizeData=np.concatenate((args[0],args[1]),axis=0)
        normalizeData=(normalizeData-normalizeData.min(axis=0))/(normalizeData.max(axis=0)-normalizeData.min(0))
        return normalizeData[:args[0].shape[0]], normalizeData[args[0].shape[0]:]
    else:
        return (args[0]-args[0].min(axis=0))/(args[0].max(axis=0)-args[0].min(0))
    

def save_embedding(savedir,savefile, *args):
    save_path = os.path.join(savedir, savefile)
    if len(args)>1:
        np.savez(save_path, emb_train=args[0],emb_test=args[1])
    else:
        np.savez(save_path, emb_train=args[0])
    
    


    