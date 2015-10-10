""" Usage: python make_submission.py csvpath model_name
csv - must contain 12320 rows, 176 coloumns: the predictions for test set
"""

import sys
import numpy as np

# info about classes
file = open('trainingData.csv')
data = file.readlines()[1:]
langs = set()
for line in data:
    filepath, language = line.split(',')
    language = language.strip()
    langs.add(language)
langs = sorted(langs)

path = sys.argv[1]
name = sys.argv[2]
read_file = open(path, 'r')
f = open('testingData.csv')
cnt = 12320
print_file = open('predictions/test_' + name + '.csv', 'w')

for iter in range(cnt):
    st = f.readline()
    name = st.strip()[:-4]
    
    out = read_file.readline().split(',')
    out = [float(x) for x in out]
    pred = sorted([(x, it) for it, x in enumerate(out)], reverse=True)

    for i in range(3):
        lang_id = pred[i][1]
        lang = langs[lang_id]
        print_file.write(name + '.mp3,' + lang + ',' + str(i + 1) + '\n')

    if (iter % 100 == 0):
        print >> sys.stderr, "processed %d / %d images" % (iter + 1, cnt)
