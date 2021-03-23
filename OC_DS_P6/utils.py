from nltk import pos_tag
from nltk.corpus import wordnet

def clean_field(col, tknzr, sw, lmtzr, stmr):
    """
    Fonction de tokenisation du contenu dont regex \w+,
    suppression des stopwords, lemmatisation et racinisation
    """

    temp = []

    for w in tknzr.tokenize(col.lower()):
        if w not in sw and not w.isdigit():
            pos = get_wordnet_pos(w)
            if pos != 'n':
                continue
            else:
                temp.append(stmr.stem(lmtzr.lemmatize(w, pos)))

    return ' '.join(temp)

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """

    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def get_tags(features, pred):
    """
    Fonction de récupératon des libellés des tags à partir de la prédiction
    """

    temp = set()

    for i in range(len(pred)):
        if pred[i] == 1:
            temp.update([features[i]])

    return '\<' + '\>\<'.join(sorted(temp)) + "\>"
