arr[1]=0.000244140625
arr[2]=0.0009765625
arr[3]=0.00390625
arr[4]=0.015625
arr[5]=0.0625
arr[6]=0.25
arr[7]=1
arr[8]=4
arr[9]=16

for var in "${arr[@]}"
do
   #time ./libsvm/svm-train -v 3 -s 0 -c "${var}" ~/Workspace/ML-HW/HW3/Problem-6/input/mod-phishing-train.txt &> ./output/result_["${var}"].txt
   { time ./libsvm/svm-train -v 3 -c ${var} \
         ./input/mod-phishing-train.txt ; } \
         1> ./output/linear/result_[${var}].txt 2> ./output/linear/result_[${var}].txt
done