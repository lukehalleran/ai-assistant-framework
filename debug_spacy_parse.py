#!/usr/bin/env python3
"""Debug spaCy parse for 'My cat's name is Flapjack'"""

import spacy

nlp = spacy.load("en_core_web_sm")

text = "My cat's name is Flapjack"
doc = nlp(text)

print(f"Text: {text}\n")
print("Token analysis:")
print(f"{'Token':<15} {'POS':<8} {'Dep':<12} {'Head':<15} {'Children'}")
print("="*80)

for token in doc:
    children = [f"{c.text}({c.dep_})" for c in token.children]
    children_str = ", ".join(children) if children else "-"
    print(f"{token.text:<15} {token.pos_:<8} {token.dep_:<12} {token.head.text:<15} {children_str}")

print("\n" + "="*80)
print("\nLooking for 'name' token:")
for token in doc:
    if token.lemma_.lower() == "name":
        print(f"  Found: '{token.text}' (pos={token.pos_}, lemma={token.lemma_})")
        print(f"  Children:")
        for child in token.children:
            print(f"    - {child.text} (dep={child.dep_}, pos={child.pos_})")
            if child.dep_ == "poss":
                print(f"      ^ This is the possessor: '{child.text}'")
                print(f"      Possessor's children:")
                for gc in child.children:
                    print(f"        - {gc.text} (dep={gc.dep_})")

print("\n" + "="*80)
print("\nDependency tree:")
from spacy import displacy
for sent in doc.sents:
    for token in sent:
        print(f"{' '*token.i*2}{token.text} --{token.dep_}--> {token.head.text}")
