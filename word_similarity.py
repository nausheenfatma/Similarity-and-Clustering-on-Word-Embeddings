# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 02:17:06 2015

@author: nausheenfatma
"""
import logging
import numpy,sys
from gensim.models import word2vec


path="text8"
msg=""

# command line arguments should be like this :
# python word_similarity.py "delhi"  
# python word_similarity.py "delhi" "islamabad" "kathmandu"
# python word_similarity.py -c "country"
# python word_similarity.py -c "city"
# python word_similarity.py -c "organization"

if sys.argv[1]=='-c':
    msg="\n\nTop words in %s class..\n "%sys.argv[2]
    if sys.argv[2]=='country':
        word_class=["india","britain","brazil"] #random country points
    elif sys.argv[2]=='city':
        word_class=["delhi","tokyo","london"] #random city points
    elif sys.argv[2]=='organization':
        word_class=["unicef","weu","un","oas"] #random organization points
    elif sys.argv[2]=='person':
        word_class=["indira","prakash","gandhi","nehru","pooja"]
else:
    msg="\n\nMost similar words.. \n"
    word_class=sys.argv[1:]

#one should use gensim  most similar method instead of this function.
#our assignment requirement was to do without using gensim
def cosine_similarity(vector1,vector2):
    cosine=numpy.dot(vector1,vector2)/(numpy.linalg.norm(vector1)*numpy.linalg.norm(vector2))
    return round(cosine,6)
    


vocab_vec_dict={}

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#####################Training Model######################################################################

sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences)
model.save(path+'model.model')
model.save_word2vec_format(path+'model.model.bin', binary=True)

#reading unique words from the corpus saved in test_vocab.txt file
f=open(path+"_vocab.txt","r")
j=0
#saving word embeddings in test_vectors file
f2=open(path+"_vectors.txt","w")
for line in f:
    word=line.rstrip()
    vocab_vec_dict[j]=[]
    vocab_vec_dict[j].append(word)
    a=(model[word])
    s=""
    for e in a:
        s=s+str(e)+" "
    s=s.rstrip(" ")
    f2.write(str(s)+"\n")
    vocab_vec_dict[j].append(s)
    j=j+1
f2.close() 

#############################finding most similar values#################################################

#vector 1 is a list variable in which word embeddings of command line arguments would be found and saved 
vector1=[]
for evry in range(len(word_class)):
    vector1.append(model[str(word_class[evry])])
    #model["abc"] fetches thw word embedding for string "abc" from the model

vector1=numpy.mean(vector1,axis=0)
similarity_array=[]
f2=open(path+"_vectors.txt","r")
i=0
for line in f2:
    i=i+1
    vector2=line.rstrip().split()
    vector2=map(float,vector2)
    similarity_array.append(cosine_similarity(vector1,numpy.array(vector2)))
f2.close()

#print model["बिहार"]

print msg
similarity_array_org=list(similarity_array)
#top_similar_values=sorted(similarity_array,reverse=True)[:no_of_words]
top_similar_values=sorted(similarity_array,reverse=True)
rank=1
for each in range(len(top_similar_values)):
    ind=similarity_array_org.index(top_similar_values[each])
    if top_similar_values[each]>=0.62:    #printing values whose similarity score is greater than 0.62
        print rank, vocab_vec_dict[ind][0],top_similar_values[each]
        rank+=1
    else:
        break
