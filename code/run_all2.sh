#!/bin/bash
for integ in 'exp+meth' 'exp+mirna' 'meth+mirna'  
 do
 for ds in 128 256 512
 do
 for lsize in 16 32 64
 do
 for distance in 'kl' 'mmd'
 do
 for beta in 1 10 15 25 50 100 
 do
 for dtype in  'OS' #whole data 
 do
 for fold in 1 2 3 4 5 6 7 8 9 10 #0 whole data
 do
  python run_xvae2.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results'
  python run_cncvae2.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results'
done
done
done
done
done
done
done
