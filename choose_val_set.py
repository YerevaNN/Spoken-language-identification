file = open('trainingData.csv')

data = file.readlines()[1:]
langs = set()
for line in data:
    filepath, language = line.split(',')
    filepath = filepath.strip()
    language = language.strip()
    langs.add(language)

langs = sorted(langs)
ID = {}
for id, lang in enumerate(langs):
    ID[lang] = id
file.close()

train = open('trainEqual.csv', 'w')
val = open('valEqaul.csv', 'w')
cnt = [0 for x in range(176)]
for line in data:
    filepath, language = line.split(',')
    filepath = filepath.strip()
    language = language.strip()
    id = ID[language]
    if (cnt[id] < 306):
        train.write(filepath[:-4] + ',' + str(ID[language]) + '\n')
    else:
        val.write(filepath[:-4] + ',' + str(ID[language]) + '\n')
    cnt[id] += 1
    
train.close()
val.close()
