""" Usage: python majority_vote_ensembling.py csv1path csv2path ..
"""
import sys
import numpy as np

n_csv = len(sys.argv) - 1
train_cnt = 12320

csv = []
for index in range(1, len(sys.argv)):
    csv.append(open(sys.argv[index], 'r'))
    
ensembled = open('top3_prediction_ensembled.csv', 'w')

for iter in range(train_cnt):
    cnt = [0 for i in range(176)]
    avg_prob = np.array([0.0 for i in range(176)])

    for index in range(n_csv):
        cur_prob = csv[index].readline().split(',')
        cur_prob = np.array([float(x) for x in cur_prob])
        
        avg_prob += cur_prob
        prediction = cur_prob.argmax()
        cnt[prediction] += 1


    mas = [(cnt[index], avg_prob[index], index) for index in range(176)]
    mas = sorted(mas, reverse=True)
    
    ensembled.write(str(mas[0][2]) + ',' + str(mas[1][2]) + ',' + str(mas[2][2]) + ',')
    ensembled.write(','.join([str(x) for x in cnt]) + '\n')
    