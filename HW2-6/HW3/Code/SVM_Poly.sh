C[4]=0.015625000000000
C[5]=0.062500000000000
C[6]=0.250000000000000
C[7]=1.000000000000000
C[8]=4.000000000000000
C[9]=16.000000000000000
C[10]=64.000000000000000
C[11]=256.000000000000000
C[12]=1024.000000000000000
C[13]=4096.000000000000000
C[14]=16384.000000000000000

degree[1]=1
degree[2]=2
degree[3]=3

for d in "${degree[@]}"
do
    for var in "${C[@]}"
    do
        { time ./libsvm/svm-train -v 3 -s 0 -c "${var}" -d "${d}" -t 1 \
         ~/Workspace/ML-HW/HW3/Problem-6/input/mod-phishing-train.txt ; } \
         1> ./output/poly/result_["${var}"]_["${d}"].txt 2> ./output/poly/result_["${var}"]_["${d}"].txt
    done
done