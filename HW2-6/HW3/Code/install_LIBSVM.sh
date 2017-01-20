wget http://www.csie.ntu.edu.tw/~cjlin/libsvm/libsvm-3.21.tar.gz
tar -zxvf libsvm-3.21.tar.gz

mv ./libsvm-3.21 ./libsvm

cd libsvm
make

cd python
make