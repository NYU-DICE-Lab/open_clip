import nltk
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_captions(x):
    import logging
    try:
        cleaned = str(x).lower().translate({ord(i): None for i in '&<^*>\\|+=[]~`\"@/\'\©#)("'}).translate({ord(i): " " for i in ':;-_.,!?\n'})\
        .replace(" www ", " ").replace(" com ", " ")\
        .replace(" flickr ", " ").replace(" st ", " street ").replace(" de ", "")\
        .replace("http", " ").replace("href", " ")\
        .strip()
        c_list = list(cleaned.split(" "))
        c_list = [c for c in c_list if (len(c) < 30 and not c.isnumeric())]
        if len(c_list) > 50:
            c_list = c_list[:49]
        return " ".join(c_list)
    except Exception as e:
        logging.info("Exception in clean captions: ")
        logging.info(e)
        return ""

def list_clean_nosplit(l):
    return set(map(clean_captions, l))

def list_clean(l):
    l = list(map(clean_captions, l))
    s = set(chain(*[s.split(" ") for s in l]))
    k = [c for c in s if c not in common]
    return k

def reverse_words(s):
    return "".join(reversed(s.split()))

def ds_val_getter(ds):
    if isinstance(ds, dict):
        ds_values = {idx:list(set([t.lower().strip() for t in row.split(", ")] + [reverse_words(t.lower().strip()) for t in row.split(", ")] + [t.lower().strip()+"s" for t in row.split(", ")] + [t.lower().strip().replace(" ", "") for t in row.split(", ")] + [reverse_words(t.lower().strip()).replace(" ", "") for t in row.split(", ")])) for idx, row in enumerate(ds.values())}
    else:
        ds_values = {idx:list(set([ds[idx].lower().strip(), reverse_words(ds[idx].lower().strip()), ds[idx].lower().strip()+"s", ds[idx].lower().strip().replace(" ", ""), reverse_words(ds[idx].lower().strip()).replace(" ", "")])) for idx in range(len(ds))}
    return ds_values

def in1k_hard_subset_match(s, ds, ngram=3, multiclass=False, strict=False):
    s = str(s)
    if s == "":
        return -1
    ds_values = ds_val_getter(ds)
    s = list(lemmatizer.lemmatize(t) for t in s.split(" "))
    grams = []
    for count, word in enumerate(s):
        for i in range(ngram):
            if count + i - 1 > len(s):
                continue
            grams.append(" ".join(w for w in s[count:count+i+1]))
            
    matches = []
    
    for gram in grams:
        for idx, val in enumerate(ds_values.values()):
            if gram in val:
                #print("Match {}, {}".format(gram, val))
                if multiclass or strict:
                    matches.append(idx)
                else:
                    return idx
    
    if matches == []:
        return -1
    elif strict and len(matches) != 1:
        return -1
    elif multiclass:
        matches = set(matches)
        return ", ".join(str(m) for m in matches)
    else:
        return matches[0]