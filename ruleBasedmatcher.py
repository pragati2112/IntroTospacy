import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en")
matcher = PhraseMatcher(nlp.vocab)
doc = nlp("German Chancellor Angela Merkel and US President Barack Obama "
          "converse in the Oval Office inside the White House in Washington, D.C.")
text=[]
for x in doc.ents:
    text.append(x.text)
print(text)
# terms = ["Barack Obama", "Angela Merkel", "Washington, D.C."]
patterns = [nlp.make_doc(t) for t in text]
matcher.add("TerminologyList", None, *patterns)


matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)
