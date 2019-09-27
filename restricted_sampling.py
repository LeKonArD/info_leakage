import pandas as pd
import re
from collections import Counter
import random
import pickle as pkl
import numpy as np
from itertools import combinations

def rmv(x):
    return re.sub("^\s+|\s+$","",x)

def clean_authors(x,pattern):
    
    if len(pattern.findall(x)) > 0:
        x = re.sub("\[.*","",x)
        x = rmv(x)
        x = x.split(",")
        try:
            x = x[1] +" "+ x[0]
        except:
            x = x[0]
    else:
        x = re.sub("\,.*","",x)
        x = rmv(x)
    x = rmv(x)
    return x


def perm(counts, sample_size):
    combs = []
    for i in range(len(counts.values()), 0, -1):
        com = combinations(counts.keys(), i)
        for seq in com:
            summe = sum([counts[x] for x in seq])
            if summe > sample_size and summe < sum(counts.values())-sample_size:
                yield seq

def shared_authors(crit, frame):
    
    aut_1 = np.unique(frame[frame.publisher.isin(list(crit))]["author"])
    aut_2 = np.unique(frame[~frame.publisher.isin(list(crit))]["author"])
    
    shared_values = [x for x in aut_1 if x not in aut_2]
    
    return shared_values

def shared_authors_reihen(crit, frame):
    
    aut_1 = np.unique(frame[frame.reihe.isin(list(crit))]["author"])
    aut_2 = np.unique(frame[~frame.reihe.isin(list(crit))]["author"])
    
    shared_values = [x for x in aut_1 if x not in aut_2]
    
    return shared_values
                
def const_strict(df, genre, sample_size):
    
    frame = df[df.label==genre]
    pubs = perm(Counter(frame.publisher), sample_size)
    for pub_comb in pubs:
                
        deletion = shared_authors(pub_comb, frame)
        
        final_crit = list(pub_comb)+list(deletion)
        
        test_crit = frame[frame.publisher.isin(final_crit)
                        & frame.author.isin(final_crit)].index.astype(str)
        
        test_crit2 = frame[~frame.publisher.isin(final_crit)
                        & ~frame.author.isin(final_crit)].index.astype(str)

        if len(test_crit) >= sample_size and len(test_crit2) >= sample_size:
 
            yield final_crit 
                
def const_author(df, genre, sample_size):
    frame = df[df.label==genre]
    authors = perm(Counter(frame["author"]), sample_size)
    for author_comb in authors:
        yield list(author_comb)
        
def const_reihe(df, genre, sample_size):
    frame = df[df.label==genre]
    reihen = perm(Counter(frame["reihe"]), sample_size)
    for reihen_comb in reihen:
        yield list(reihen_comb)
        
def const_publisher(df, genre, sample_size):
    frame = df[df.label==genre]
    pubs = perm(Counter(frame.publisher), sample_size)
    for pubs_comb in pubs:
        yield list(pubs_comb)

def const_autpub(df, genre, sample_size):
    
    frame = df[df.label==genre]
    pubs = perm(Counter(frame.publisher), sample_size)
    
    for pub_comb in pubs:
            
        authors = perm(Counter(frame[frame.publisher.isin(list(pub_comb))]["author"]), sample_size)
        
        for author_comb in authors:
            
            yield [x for x in list(author_comb)+list(pub_comb)]
    
def const_reipub(df, genre, sample_size):
    
    frame = df[df.label==genre]
    pubs = perm(Counter(frame.publisher), sample_size)
    
    for pub_comb in pubs:
            
        reihen = perm(Counter(frame[frame.publisher.isin(list(pub_comb))]["reihe"]), sample_size)
        
        for reihe_comb in reihen:
            
            yield [x for x in list(reihe_comb)+list(pub_comb)]
    
def const_reiaut(df, genre, sample_size):
    
    frame = df[df.label==genre]
    reihen = perm(Counter(frame.reihe), sample_size)
    for reihe_comb in reihen:
                
        deletion = shared_authors_reihen(reihe_comb, frame)
        
        final_crit = list(reihe_comb)+list(deletion)
        
        test_crit = frame[frame.reihe.isin(final_crit)
                        & frame.author.isin(final_crit)].index.astype(str)
        
        test_crit2 = frame[~frame.reihe.isin(final_crit)
                        & ~frame.author.isin(final_crit)].index.astype(str)

        if len(test_crit) >= sample_size and len(test_crit2) >= sample_size:
 
            yield final_crit 
            
            
def constrain(df, genre, sample_size, case):
    
    pattern = re.compile("\[")
    
    df = df.drop(df[df.author.astype(str).isin(["0","nan"])].index)
    df = df.drop(df[df.reihe.astype(str).isin(["0","nan"])].index)
    df = df.drop(df[df.publisher.astype(str).isin(["0","nan"])].index)
    df["author"] = df["author"].apply(lambda x: clean_authors(x,pattern))
    df_focus = df[df.label == genre]
    df_counter = df[df.label != genre]
    
    if case == "author":

        g = const_author(df, genre, sample_size)
        restrictions = next(g)
        
        focus_train = df_focus[df_focus.author.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)[:sample_size]
        
        restrictions = np.unique(df_focus.loc[focus_train,"author"])
        
        focus_test = df_focus[~df_focus.author.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)
       
        
    if case == "reihe":

        g = const_reihe(df, genre, sample_size)
        restrictions = next(g)
        
        focus_train = df_focus[df_focus.reihe.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)[:sample_size]
        
        restrictions = np.unique(df_focus.loc[focus_train,"reihe"])
        
        focus_test = df_focus[~df_focus.reihe.isin(restrictions)].index.astype(str)

    if case == "publisher":

        g = const_publisher(df, genre, sample_size)
        restrictions = next(g)

        focus_train = df_focus[df_focus.publisher.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)[:sample_size]
        
        restrictions = np.unique(df_focus.loc[focus_train,"publisher"])
        
        focus_test = df_focus[~df_focus.publisher.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)
    
    
    if case == "autpub":
        
        g = const_autpub(df, genre, sample_size)
        restrictions = next(g)
        
        focus_train = df_focus[df_focus.publisher.isin(restrictions) 
                          & df_focus.author.isin(restrictions)].index.astype(str)
        
        restrictions = np.unique(list(df_focus.loc[focus_train,"publisher"])+list(df_focus.loc[focus_train,"author"]))
        
        
        focus_test = df_focus[~df_focus.publisher.isin(restrictions)
                          & ~df_focus.author.isin(restrictions)].index.astype(str)
    
    
    if case == "reipub":
        
        g = const_reipub(df, genre, sample_size)
        restrictions = next(g)
        
        focus_train = df_focus[df_focus.publisher.isin(restrictions) 
                          & df_focus.reihe.isin(restrictions)].index.astype(str)
        
        restrictions = np.unique(list(df_focus.loc[focus_train,"publisher"])+list(df_focus.loc[focus_train,"reihe"]))
        
        
        focus_test = df_focus[~df_focus.publisher.isin(restrictions)
                          & ~df_focus.reihe.isin(restrictions)].index.astype(str)
    
    if case == "reiaut":
        
        g = const_reiaut(df, genre, sample_size)
        restrictions = next(g)
        
        focus_train = df_focus[df_focus.reihe.isin(restrictions)
                          & df_focus.author.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)
        
        restrictions = np.unique(list(df_focus.loc[focus_train,"author"])+list(df_focus.loc[focus_train,"reihe"]))
        
        focus_test = df_focus[~df_focus.reihe.isin(restrictions)
                          & ~df_focus.author.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str) 

    
    if case == "strict":
        
        g = const_strict(df, genre, sample_size)
        restrictions = next(g)
        
        focus_train = df_focus[df_focus.publisher.isin(restrictions) 
                             & df_focus.author.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)

        restrictions = np.unique(list(df_focus.loc[focus_train,"publisher"])+list(df_focus.loc[focus_train,"author"]))
        
        focus_test = df_focus[~df_focus.publisher.isin(restrictions)
                            & ~df_focus.author.isin(restrictions)].sample(frac=1, random_state=42).index.astype(str)
    
    

        
    if case == "random":
        
        focus_train = list(df_focus.sample(frac=1, random_state=42).index.astype(str))[:sample_size]
        focus_test = list(df_focus[~df_focus.index.isin(focus_train)].sample(frac=1, random_state=42).index.astype(str))
   
    


    focus_train = focus_train[:sample_size]
    
    genre_count = int(sample_size/11)+1
    counter_train = list(df_counter.sample(frac=1, random_state=42).groupby('label', group_keys=False).apply(lambda x: x.sample(genre_count, random_state=42)).index.astype(str))[:sample_size]
    
    genre_count = int(len(focus_test)/(len(np.unique(df.label))))+1  
    counter_test = list(df_counter.sample(frac=1, random_state=42)[~df_counter.index.isin(focus_train)].groupby('label', group_keys=False).apply(lambda x: x.sample(genre_count, random_state=42)).index.astype(str))[:len(focus_test)]
    
    X_train = list(focus_train) + list(counter_train)
    X_test = list(focus_test) + list(counter_test)
    y_train = len(focus_train)*[0]+len(counter_train)*[1]
    y_test = len(focus_test)*[0]+len(counter_test)*[1]
    if len(X_train) < sample_size*2 or len(X_test) < sample_size*2:
        return [0],[0],[0],[0],"0","0"
    
    return X_train, X_test, y_train, y_test, case, genre

        
def get_constrained_data(genre, case, sample_size, df, metadata):
    '''
    input:
      genre: str
      Testgenre: z.b. Horror
    
      case: str
      Gibt den constaint an (strict, random)
    
      sample_size: str
      Mindestanzahl der Trainingsbeispiele für die Fokusklasse
    
      df: pandas.DataFrame
      DocumentTermMatrix
    
    output:
      analog zu sklearn.model_selection.train_test_split
    
    '''
    try:
        X_train, X_test, y_train, y_test, case, genre = constrain(metadata, genre, sample_size, case)
    except StopIteration:
        print("Consraints werden gelockert")
        case = "strict+"
        try:
            
            X_train, X_test, y_train, y_test, case, genre = constrain(metadata, genre, sample_size, "reiaut")
            case = "strict+"
            
        except StopIteration:
            print("Kombination nicht möglich")
            return [None], None, None, None, "strict+"
            
    if X_train == [0]:
        print("Kombination nicht möglich")
        return [None], None, None, None, "strict+"
    
    X_train = np.array(df.T.loc[X_train,:])
    X_test = np.array(df.T.loc[X_test,:])    
    
    return X_train, X_test, y_train, y_test , case
