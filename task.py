import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
train_file = r'news_train.txt'
test_file = r'news_test.txt'
train_texts = []
train_category = []
data = list(csv.reader(open(train_file, 'rt', encoding="utf8"), delimiter='\t'))

for value in data:
    train_category.append(value[0])
    train_texts.append(value[2])

vec = TfidfVectorizer()
train_texts = vec.fit_transform(train_texts)

cl = LogisticRegression(C=7, solver='lbfgs')
cl.fit(train_texts, train_category)

data = list(csv.reader(open(test_file, 'rt', encoding="utf8"), delimiter='\t'))
test_news = []

for value in data:
    test_news.append(value[1])

i = 0
predict = []

for value in test_news:
    i += 1
    test_texts = vec.transform([value]).toarray()
    ans = str(cl.predict(test_texts))
    predict.append(ans[2:len(ans) - 2])

with open('output.txt', 'w+') as output:
    output.write('\n'.join(predict))
