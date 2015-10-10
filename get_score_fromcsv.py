""" Usage: python get_score_fromcsv.py csv_file_path 
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

csv = open(sys.argv[1], 'r')
f = open('valEqual.csv')
cnt = 12320
topcoder_score = 0.0

for iter in range(cnt):
    st = f.readline()
    (name, label) = st.split(',')
    label = int(label)

    out = csv.readline().split(',')
    out = [float(x) for x in out]
    pred = sorted([(x, it) for it, x in enumerate(out)], reverse=True)

    if (pred[0][1] == label):
        topcoder_score = topcoder_score + 1000
    elif (pred[1][1] == label):
        topcoder_score = topcoder_score + 400
    elif (pred[2][1] == label): 
        topcoder_score = topcoder_score + 160

    if (iter % 100 == 0):
        print >> sys.stderr, "processed %d / %d images" % (iter + 1, cnt)
        print >> sys.stderr, "expected score:", topcoder_score / (iter + 1) * 35200

print >> sys.stderr, "Final score: ", topcoder_score, " / ", cnt, "000"
print >> sys.stderr, "expected score:", topcoder_score / cnt * 35200
