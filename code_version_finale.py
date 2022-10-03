#third version

import spacy
from spacy.language import Language
from spacy.lang.fr import French
import numpy as np
import pandas as pd
from collections import Counter

nlp = spacy.load('fr_core_news_sm')

#load the corpus and pre-treatment
#with open('/Users/Yuke/Desktop/Master_Langue_et_informatique/memoire/claire_de_lune_maupassant.txt', 'w') as f:
#    with open('/Users/Yuke/Desktop/Master_Langue_et_informatique/memoire/claire_de_lune_maupassant_brut.txt', 'r') as fp:
#        for line in fp:
#            line = str(line).replace("\n", " ")
#            f.write(line)
#f.close()
#fp.close()

f=open("/Users/Yuke/Desktop/Master_Langue_et_informatique/memoire/programme/claire_de_lune_maupassant.txt")
text=f.read()
document=nlp(text)

phrasesList = list(document.sents)
negationsList=['pas','jamais','guère','rien','sans', 'ne', 'ni',"n'", 'personne', 'point', 'aucun', 'non']
pronoList=['se','me','te']
negVar='neg'
affVar='aff'
phrasesDict={}

for phrase in phrasesList:
    phraseStr=str(phrase)
    phrasesDict[phraseStr]=['']*7
    phrasesDict[phraseStr][1] = [] 
    phrasesDict[phraseStr][2] = []
    phrasesDict[phraseStr][3]= affVar
    phrasesDict[phraseStr][4] = []
    phrasesDict[phraseStr][5] = []
    phrasesDict[phraseStr][6] = []
    #phrasesDict.keys() = phrase
    #phrasesDict[phraseStr][0] = prono
    #phrasesDict[phraseStr][1] = aff_verbs
    #phrasesDict[phraseStr][2] = aff_lemmas
    #phrasesDict[phraseStr][3] = phrases_type
    #phrasesDict[phraseStr][4] = type_negations
    #phrasesDict[phraseStr][5] = neg_verbs
    #phrasesDict[phrasesStr][6] = neg_lemmas
  
    for word in phrase:
        if word.lemma_ in pronoList:
            phrasesDict[phraseStr][0]=str(word)
            xProno = word.i  
        elif str(word).lower() in negationsList:
            phrasesDict[phraseStr][3]=negVar
            phrasesDict[phraseStr][4].append(str(word))
            xN = word.i
        elif word.tag_ == 'VERB':                
            if phrasesDict[phraseStr][0] !='' and word.i-xProno <= 2 and phrasesDict[phraseStr][3]==affVar:
                phrasesDict[phraseStr][1].append(phrasesDict[phraseStr][0] + " " + str(word))
                phrasesDict[phraseStr][2].append('se'+' '+ word.lemma_) 
            elif phrasesDict[phraseStr][0] !='' and word.i-xProno > 2 and phrasesDict[phraseStr][3]==affVar:
                phrasesDict[phraseStr][1].append(str(word))
                phrasesDict[phraseStr][2].append(word.lemma_)
            elif phrasesDict[phraseStr][0] =='' and phrasesDict[phraseStr][3]==affVar:
                phrasesDict[phraseStr][1].append(str(word))
                phrasesDict[phraseStr][2].append(word.lemma_)
            elif phrasesDict[phraseStr][0] !='' and word.i-xProno <= 2 and phrasesDict[phraseStr][3]==negVar\
            and abs(xN-word.i)<=5:
                phrasesDict[phraseStr][5].append(phrasesDict[phraseStr][0] + " " + str(word))
                phrasesDict[phraseStr][6].append('se'+' '+ word.lemma_) 
            elif phrasesDict[phraseStr][0] !='' and word.i-xProno <= 2 and phrasesDict[phraseStr][3]==negVar\
            and abs(xN-word.i)>5:
                phrasesDict[phraseStr][1].append(phrasesDict[phraseStr][0] + " " + str(word))
                phrasesDict[phraseStr][2].append('se'+' '+ word.lemma_) 
            elif phrasesDict[phraseStr][0] =='' and phrasesDict[phraseStr][3]==negVar and abs(xN-word.i)<=5:
                phrasesDict[phraseStr][5].append(str(word))
                phrasesDict[phraseStr][6].append(word.lemma_)
            elif phrasesDict[phraseStr][0] =='' and phrasesDict[phraseStr][3]==negVar and abs(xN-word.i)>5:
                phrasesDict[phraseStr][1].append(str(word))
                phrasesDict[phraseStr][2].append(word.lemma_)
            else:
                phrasesDict[phraseStr][1].append(str(word))
                phrasesDict[phraseStr][2].append(word.lemma_)
    for word in phrase:
        if word.tag_ == 'AUX' and len(phrasesDict[phraseStr][2]) == 0:
            if phrasesDict[phraseStr][3] == affVar:
                phrasesDict[phraseStr][1].append(str(word))
                phrasesDict[phraseStr][2].append(word.lemma_) 
            elif phrasesDict[phraseStr][3] ==negVar and abs(xN-word.i)<=5:
                phrasesDict[phraseStr][5].append(str(word))
                phrasesDict[phraseStr][6].append(word.lemma_)
            elif phrasesDict[phraseStr][3] ==negVar and abs(xN-word.i)>5:  
                phrasesDict[phraseStr][1].append(str(word))
                phrasesDict[phraseStr][2].append(word.lemma_)
                
                
phrasesDict = \
{key : value for key, value in phrasesDict.items() if len(value[1]) != 0 and value[1][0].isalpha()}
    
#print (phrasesDict)


# draw table
keys=[key for key in phrasesDict.keys()]
keys=np.array(keys).reshape(-1,1)
values=[value for value in phrasesDict.values()]
values=np.array(values).reshape(-1,7)
result=np.concatenate((keys,values),axis=1)
result_df=pd.DataFrame(result,columns=['phrases','prono','aff_verbes','aff_lemmas','types_phrases',\
                                       'type_negations', 'neg_verbs', 'neg_lemmas'])
#print (result_df.head())

#write to locality in .csv
#result_df.to_csv('/Users/Yuke/Desktop/result.csv')


#create the table of the analyse result
# extract indicated columns
result_analyse=result_df.loc[:,['aff_lemmas',"neg_lemmas"]]

# extract indicated lines from indicated elements' values 
types_aff=result_analyse["aff_lemmas"]
types_neg=result_analyse["neg_lemmas"]

# calculate appearence times for each element and record them in the renamed column
lemmas_aff = []
for lemmas in types_aff:
    for lemma in lemmas:
        lemmas_aff.append(lemma)
counts_aff = pd.DataFrame.from_dict(dict(Counter(lemmas_aff)), orient='index').reset_index()
lemmas_neg = []
for lemmas in types_neg:
    for lemma in lemmas:
        lemmas_neg.append(lemma)
counts_neg = pd.DataFrame.from_dict(dict(Counter(lemmas_neg)), orient='index').reset_index()
counts_aff.columns = ['lemmas', 'counts_aff']
counts_neg.columns=['lemmas','counts_neg']

# merge two frames on indicated elements
result_analyse=pd.merge(counts_aff, counts_neg, how='outer', on=['lemmas'])

# replace the value 'nan' by 0
result_analyse = result_analyse.fillna(0)

# add a line by indicated index and calculate the sum of indicated colunm
result_analyse.loc[len(result_analyse)]= \
['total',result_analyse['counts_aff'].sum(),result_analyse['counts_neg'].sum()]

# add a new colunm filled values by 'nan'
result_analyse['counts'], result_analyse['percentage_aff'], result_analyse['percentage_neg'],\
result_analyse['probability_aff'],result_analyse['probability_neg'], \
result_analyse['preference']= [np.nan, np.nan, np.nan,np.nan,np.nan, np.nan]

# calculate elements from some colunms and add values to indicated column
result_analyse['counts'] = result_analyse.apply(lambda x: x['counts_aff'] + x['counts_neg'], axis=1)
result_analyse['percentage_aff'] = \
(result_analyse['counts_aff'] / result_analyse['counts']).apply(lambda x: '%.2f%%' % (x*100))
result_analyse['percentage_neg'] = \
(result_analyse['counts_neg'] / result_analyse['counts']).apply(lambda x: '%.2f%%' % (x*100))
result_analyse['probability_aff'] = \
(result_analyse['counts_aff'] / (result_analyse.iloc[len(result_analyse)-1,1]))
result_analyse['probability_neg'] = \
(result_analyse['counts_neg'] / (result_analyse.iloc[len(result_analyse)-1,2]))
result_analyse['preference'] = \
(result_analyse['probability_aff'] - result_analyse['probability_neg'])\
.apply(lambda x: '%.2f%%' % (x*100)) 
result_analyse['probability_aff'] = \
(result_analyse['counts_aff'] / (result_analyse.iloc[len(result_analyse)-1,1])).apply(lambda x: '%.2f%%' % (x*100)) 
result_analyse['probability_neg'] = \
(result_analyse['counts_neg'] / (result_analyse.iloc[len(result_analyse)-1,2])).apply(lambda x: '%.2f%%' % (x*100))

#change places of indicated colunms
result_analyse = result_analyse[['lemmas','counts','counts_aff','counts_neg','percentage_aff',\
                                'percentage_neg','probability_aff','probability_neg','preference']]


#print (result_analyse)
#pd.set_option('display.max_rows', None)
#result_analyse.to_csv('/Users/Yuke/Desktop/result_analyse.csv')
    
