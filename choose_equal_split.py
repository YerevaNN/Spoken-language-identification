"""split data into training and validation sets"""
import csv

with open('trainingData.csv', 'rb') as csvfile:
    next(csvfile) #skip headers
    data = list(csv.reader(csvfile, delimiter=','))

    #Map every language to an ID
    langs = set([language.strip() for _,language in data])
    ID = {lang: i for i,lang in enumerate(sorted(langs))}

    #Write first 306 items to training set and the rest to validation set
    cnt = [0 for _ in range(len(langs))]
    with open('trainEqual.csv', 'w') as train:
        with open('valEqaul.csv', 'w') as val:
            for line in data:
                filepath, language = map(str.strip, line)
                id_lang = ID[language]

                if (cnt[id_lang] < 306):
                    train.write(filepath[:-4] + ',' + str(id_lang) + '\n')
                else:
                    val.write(filepath[:-4] + ',' + str(id_lang) + '\n')
                cnt[id_lang] += 1
