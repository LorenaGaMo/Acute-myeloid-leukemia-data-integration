for integ in 'exp+meth' 'exp+mirna' 'meth+mirna'
do
   python analyse_representations2_512.py --integration=${integ} --model='CNCVAE' --dtype='W' --resdir='results' --writedir='tsne_Analyses_512'
   python analyse_representations2.py --integration=${integ} --model='CNCVAE' --dtype='W' --resdir='results' --writedir='tsne_Analyses' 
done

