import json
import numpy as np
import pandas as pd
import os

"""
Stage: Stage 1 file

Document type: Data preprocessing for extracting embedding feature and semantic feature only.

Main purpose: 1. To extract the word embedding feature and semantic feature for each sentence

Need to run? No.

Dependency:
    use -> None
    be used -> None

Methods:
    load_embedding(): To load the Glove300 embeddings
    remove_stop_word(stop,text): Remove the stop words from the stopword lexicon
    ind_semantic(file_open_path : str, dict_semantic : dict) -> list : Remove the stop words and get the array of semantics
    semantic_dictionary_as_vector(): helper-function: The main effect is to make the semantic dictionary as the dict variable to improve the speed
    find_word_vectorization(text, file_output_path:str,embedding : dict): sum each word's vector up to represent the sentence
    main(): To extract all files' word embeddings and semantic vector
"""

def load_embedding():
    """
    To load the Glove300 embeddings
    :return:
    """
    embeddings_index = {}
    f = open('glove.840B.300d.txt')
    a=0
    for line in f:
        a+=1
        values = line.split(' ')
        word = values[0] ## The first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
        embeddings_index[word] = coefs
        if(a%50000 == 0):
            print(a)
    f.close()

    print('GloVe data loaded')
    return embeddings_index
#Stop: the path to the stop words in txt.format
#Text: json file as a list
def remove_stop_word(stop,text):
    """
    Remove the stop words from the stopword lexicon
    :param stop: The path of stopword lexicon
    :param text: The text path for the input
    :return: the text after removing the stopword
    """
    import re

    ## Iterate over the data to preprocess by removing stopwords
    t = open(stop,'r')
    t = t.read()
    t = t.split("\n")
    print(t)
    output = text
    for line in output:
        try:
            line.append(line[3].lower())
            #print(line[3])
        except:
            continue
        #line_by_words = re.findall(r'(?:\w+)', line[3], flags = re.UNICODE) # remove punctuation ans split
        line_by_words = re.findall(r'(?:[A-Za-z]+)', line[3], flags = re.UNICODE)
        new_line=[]
        for word in line_by_words:
            #for words not in the stop words
            if word not in t:
                new_line.append(word)
        line[4] = new_line
        #print(lines_without_stopwords)
    #texts = lines_without_stopwords
    print(output)
    return output
    #print(texts)

#a = json.load(open("SP500SentenceList/A/4223396.json","r"))
#remove_stop_word(0,a[2])

#output:-> we will have 5 things. (num,speaker,"P/Q/A",sentence,sentence without stopwords, semantics)
def find_semantic(file_open_path : str, dict_semantic : dict) -> list :
    '''
    Remove the stop words and get the array of semantics
    :param file_open_path: The open path of original file
    :param dict_semantic: the dictionsry for semantics
    :return: the text with array of semantics
    '''
    text =  remove_stop_word("stopwords-en-master/stopwords-en.txt",
                             json.load(open(file_open_path,"r")))
    #print(f)
    dict = dict_semantic
    num=0
    delete = list()
    for each_sentence in text:
        #'negative,positive,uncertainty,litigious,strong_modal,weak_modal,constraining,complexity
        sen_semantic = np.array([0,0,0,0,0,0,0,0])
        if each_sentence[3] != None:
            each_sentence[0] = num
            for each_word in each_sentence[4]:
                word = each_word.upper()
                if word in dict.keys():
                    sen_semantic += dict[word]

            each_sentence.append(sen_semantic.tolist())
            num+=1
        else:
            delete.append(each_sentence)
    for each in delete:
        text.remove(each)

    #print(text)
    #json.dump(text,open(file_write_path,"w"))
    return text

def semantic_dictionary_as_vector() -> dict:
    '''
    helper-function:
    The main effect is to make the semantic dictionary as the dict variable to improve the speed
    :return: dict
    '''
    n = 0
    f = pd.read_excel('Updated_LoughranMcDonald_MasterDictionary_2020.xlsx')
    output = dict()
    for each in f.iterrows():
        output[each[1]['Word']] = np.array([0,0,0,0,0,0,0,0])
        if each[1]['Negative'] != 0:
            output[each[1]['Word']][0] = 1
        if each[1]['Positive'] != 0:
            output[each[1]['Word']][1] = 1
        if each[1]['Uncertainty'] != 0:
            output[each[1]['Word']][2] = 1
        if each[1]['Litigious'] != 0:
            output[each[1]['Word']][3] = 1
        if each[1]['Strong_Modal'] != 0:
            output[each[1]['Word']][4] = 1
        if each[1]['Weak_Modal'] != 0:
            output[each[1]['Word']][5] = 1
        if each[1]['Constraining'] != 0:
            output[each[1]['Word']][6] = 1
        if each[1]['Complexity'] != 0:
            output[each[1]['Word']][7] = 1
        n += 1
        if n % 100 == 1:
            print(n)
    return output


#find_semantic()
#c = remove_stop_word(0,json.load(open("SP500SentenceListAfterClean0402/A/4223396.json","r")))
#print(len(c))

def find_word_vectorization(text, file_output_path:str,embedding : dict):
    '''
    sum each word's vector up to represent the sentence
    :param text: The transition list from the semantic method
            file_output_path: the output path
            embedding: the vector embedding
    :return:
    '''
    embedding = embedding
    for each_sentence in text:
        sum = [0 for i in range(0,300)]
        for each_word in each_sentence[4]:
            try:
                sum+= embedding[each_word]
            except:
                continue
        each_sentence.append(list(sum))
    print(text)
    json.dump(text,open(file_output_path,"w"))
    return text

#find_word_vectorization(json.load(open("SP500_stopword_semantics/4223396.json","r")))

def main():
    """
    To extract all files' word embeddings and semantic vector
    :return:
    """
    dictionary_semantic = semantic_dictionary_as_vector()
    dictionary_vectorization = load_embedding()
    os.chdir("SP500SentenceListAfterClean0402")
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            file = str(os.path.join(root, name))[1:]
            filename = "SP500SentenceListAfterClean0402" + file
            find_word_vectorization(find_semantic(filename, dictionary_semantic),filename, dictionary_vectorization)


