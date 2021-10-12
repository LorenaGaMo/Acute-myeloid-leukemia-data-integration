for integ in 'exp+meth' 'exp+mirna' 'meth+mirna'
do
   python analyse_representationsCol.py --integration=${integ} --model='XVAE' --dtype='W' --resdir='results' --writedir='tsne_AnalysesCol'
done

