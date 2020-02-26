import spacy
from spacy.matcher import PhraseMatcher
# from pathlib import path
# import plac
import random

nlp = spacy.load('en')
# doc = nlp(u"hello Thomas, I've some good news to give you in London.")
# cleaned = []
# for y in doc:
#     if not y.is_stop and y.pos_!="PUNCT":
#         cleaned.append(y)
# print(cleaned)
# raw = [(x.lemma_,x.pos_)for x in cleaned]
# print(raw)
doc= nlp('The copper statue, a gift from the people of France to the people of the United States')
print(doc.ents)
ents=[(x.text, x.label_) for x in doc.ents]
print(ents)
if 'ner' not in nlp.pipe_names:
    ner=nlp.create_pipe('ner')
    nlp.add_pipe('ner')
else:
    ner=nlp.get_pipe('ner')

# 56
matcher=PhraseMatcher(nlp.vocab)
# for i in ['GINA HASPEL','HASPEL','GINA']:
#     matches=matcher.add(label,None,nlp(i))
# for match in matches:
#     print(match)


res=[]
train_ents=[]
matches=matcher(doc)
for x in doc.ents:
    label=x.label_
    text= x.text
    print(text)
    print(label)


def offseter(label, doc, matchitem):
    o_one = len(str(doc[0:matchitem[1]]))
    print("hey")
    subdoc = doc[matchitem[1]:matchitem[2]]
    o_two = o_one + len(str(subdoc))
    return (o_one, o_two, label)


for l in label:
    print('hii')
    res=[offseter(label,doc,x) for x in matches]
    train_ents.append((doc,dict(entities=res)))
print(train_ents)


optimizer=nlp.begin_training()
other_pipes=[pipe for pipe in nlp.pipe_names if pipe!='ner']
with nlp.disable_pipes(*other_pipes):
    for itn in range(20):
        losses={}
        random.shuffle(train_ents)


