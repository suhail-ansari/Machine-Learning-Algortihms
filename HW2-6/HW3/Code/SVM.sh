mkdir ./output
mkdir ./output/linear
mkdir ./output/poly
mkdir ./output/rbf

python ./gen_LIBSVM_files.py

echo ">> Running Linear SVM"
bash ./SVM_Linear.sh
echo ">> Running Polynomial Kernel SVM"
bash ./SVM_Poly.sh
echo ">> Running RBF Kernel SVM"
bash ./SVM_RBF.sh

python ./get_data_from_files.py