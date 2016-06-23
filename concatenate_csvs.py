""" Usage: python concatenate_csvs.py csv1path csv2path ..
"""
import sys
import numpy as np

n_csv = len(sys.argv) - 1
cnt = 12320

csv = []
for index in range(1, len(sys.argv)):
    csv.append(open(sys.argv[index], 'r'))
    
outfile = open('concatenated.csv', 'w')

for iter in range(12320):
    out = []
    for index in range(n_csv):
        cur_out = csv[index].readline().split(',')
        cur_out = [float(x) for x in cur_out]
        out += cur_out

    out = [("%.6f" % x) for x in out]
    outfile.write(','.join(out) + '\n')