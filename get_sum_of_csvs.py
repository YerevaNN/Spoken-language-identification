""" Usage: python get_sum_csvs.py csv1path csv2path ..
"""
import sys
import numpy as np

n_csv = len(sys.argv) - 1
cnt = 12320

csv = []
for index in range(1, len(sys.argv)):
    csv.append(open(sys.argv[index], 'r'))
    
combined = open('combined.csv', 'w')

for iter in range(12320):
    out = np.zeros((176,), dtype=np.float32)
    for index in range(n_csv):
        cur_out = csv[index].readline().split(',')
        cur_out = [float(x) for x in cur_out]
        out += cur_out
    
    out = [str(x) for x in out]
    combined.write(','.join(out) + '\n')
