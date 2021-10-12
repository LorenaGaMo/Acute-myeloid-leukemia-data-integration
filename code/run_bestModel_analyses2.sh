for integ in 'exp+meth' 'exp+mirna' 'meth+mirna'
 do
 for model in 'XVAE' 'CNCVAE' 'BENCH'
 do
 for label in 'OS' 
 do
  python analyse_representations2.py --integration=${integ} --model=${model} --dtype=${label} --numfolds=10 --resdir='results' --writedir='BestModel_Analyses' --NB='True' --SVM='True' --RF='True'
done
done
done

