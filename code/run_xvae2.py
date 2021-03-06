import numpy as np
import argparse
import os

from models.xvae2 import XVAE
from misc.dataset2 import Dataset, DatasetWhole
from misc.helpers1 import normalizeRNA,save_embedding


configs = {
    128: {
 
        'ds': 128,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2

    },
    256: {
 
        'ds': 256,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },
    512: {

        'ds': 512,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--integration', help='Type of integration exp+meth, exp+mirna or meth+mirna', type=str, required=True, default='exp+meth')
parser.add_argument('--ds', help='The intermediate dense layers size', type=int, required=True)
parser.add_argument('--save_model', help='Saves the weights of the model', action='store_true')
parser.add_argument('--fold', help='The fold to train on, if 0 will train on the whole data set', type=str, default='0')
parser.add_argument('--dtype', help='The type of data (OS,W)', type=str, default='OS')
parser.add_argument('--beta', help='beta size', type=int, required=True)
parser.add_argument('--distance', help='regularization', type=str, default='kl')
parser.add_argument('--ls', help='latent dimension size', type=int, required=True)
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')



if __name__ == "__main__":
    args = parser.parse_args()
    config = configs[args.ds]
    for key, val in config.items():
        setattr(args, key, val)
 
    
if (args.fold == '0'): # whole data set
    
    print('TRAINING on the complete data')
    
    dataset_i = DatasetWhole('W')

    if (args.integration == 'exp+meth'): #integrate exp+meth
        s1_train = normalizeRNA( dataset_i.train['expnp'] )
        s2_train = normalizeRNA(dataset_i.train['methnp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]

        
        
    elif (args.integration == 'exp+mirna'): #integrate exp+mirna
        s1_train = normalizeRNA(dataset_i.train['expnp'])
        s2_train = normalizeRNA(dataset_i.train['mirnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]

        
    else:#integrate meth+mirna
        s1_train = normalizeRNA(dataset_i.train['methnp']) 
        s2_train = normalizeRNA(dataset_i.train['mirnanp'])    
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]



    xvae = XVAE(args)
    xvae.build_model()


    xvae.train(s1_train, s2_train, s1_train, s2_train)
    emb_train = xvae.predict(s1_train, s2_train)

    if (args.writedir == ''):
        emb_save_dir = 'results/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    else:
        emb_save_dir = args.writedir+'/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    emb_save_file = args.dtype +'.npz'
    save_embedding(emb_save_dir,emb_save_file,emb_train)
    
else:

    print('TRAINING on the fold '+ format(args.fold))
    
    dataset = Dataset(args.dtype, args.fold)

    if (args.integration == 'exp+meth'): #integrate exp+meth
        s1_train = normalizeRNA(dataset.train['expnp']) 
        s1_test = normalizeRNA(dataset.test['expnp'])
        s2_train, s2_test = normalizeRNA(dataset.train['methnp'],dataset.test['methnp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]


        
    elif (args.integration == 'exp+mirna'): #integrate exp+mirna
        s1_train = normalizeRNA(dataset.train['expnp']) 
        s1_test = normalizeRNA(dataset.test['expnp'] )        
        s2_train = normalizeRNA (dataset.train['mirnanp'])
        s2_test = normalizeRNA(dataset.test['mirnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]

    
    else:#integrate meth+mirna
        s1_train = normalizeRNA(dataset.train['methnp'])
        s1_test = normalizeRNA(dataset.test['methnp'])  
        s2_train, s2_test = normalizeRNA(dataset.train['mirnanp'],dataset.test['mirnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]



    xvae = XVAE(args)
    xvae.build_model()

  
    xvae.train(s1_train, s2_train, s1_test, s2_test)
    emb_train = xvae.predict(s1_train, s2_train)
    emb_test = xvae.predict(s1_test, s2_test)
    
    if (args.writedir == ''):
        emb_save_dir = 'results/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    else:
        emb_save_dir = args.writedir+'/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    emb_save_file = args.dtype +args.fold+'.npz'
    save_embedding(emb_save_dir,emb_save_file,emb_train, emb_test)
    
    