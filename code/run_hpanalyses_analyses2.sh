for integ in 'exp+meth' 'exp+mirna' 'meth+mirna'
 do
 for model in  'XVAE'  'CNCVAE'
 do
  python analyse_representations2.py --integration=${integ} --model=${model} --dtype='OS' --numfolds=10 --resdir='results' --writedir='HParameters_Analyses2'
done
done
