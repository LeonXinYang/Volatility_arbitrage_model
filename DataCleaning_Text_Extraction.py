import json
import ssl
import time
import random
import zlib
import os
from bs4 import BeautifulSoup
import socket
import DataCollection_text_collection
"""
Stage: Stage 1 file

Document type: Data preprocessing only.

Main purpose: 1. To extract the text from the json file, because the original json file is in a messy.

Need to run? No.

Dependency:
    use -> None
    be used -> None

Methods:
    take_article_content(company, document_name): To get the content from the article
    article_with_audio()-> list: To check whether the article is with corresponding audio.  (Only select the article with audio)
    beautifulsoup(str): analysis the content from the output of take_article_content
    main():Complete the extraction and analysis from json file using the beautiful soup.
    
"""


"""
This document is first to get the article with the audio already.
Secondly,we use the beautiful soup to parse the document 
Thirdly, we split each sentences in the document as such format:
    (sentence_num, speaker, Q or A, content)
Finally, we store them in the local as json
"""
def take_article_content(company, document_name):
    """
    To get the transcript from the article
    :param company: company number
    :param document_name: document number
    :return: the output as json file
    """
    #the following line should replace with company and document_name
    str = "JsonSp500/"+company+"/"+document_name+'.json'
    js = json.load(open(str))['data']['attributes']['content'];
    return js

#This method is to get the list of article with audio
#output -> (represent,question,answer)
def article_with_audio()-> list:
    """
    To check whether the article is with corresponding audio.
    (Only select the article with audio)
    :return: The list of document number with audio
    """
    output = []
    sp500audio = os.listdir("/Volumes/My Passport/Research Data/SP500Audio/")
    for each in sp500audio[1:]:
        str = "/Volumes/My Passport/Research Data/SP500_stopword_semantics/"+each+"/"
        for audio in os.listdir(str):
            output.append((each,audio[:-4]))
    return output

def beautifulsoup(str):
    """
    Analyze the sentence in the json file
    :param str: string to be analyzed
    :return: the result after analysis
    """
    from bs4 import BeautifulSoup as bs
    #(num, speaker, P&Q&A, sentence)
    output = []
    #represent = []
    #question = []
    #answer = []
    num = 0
    speaker = None
    o = bs(str,'html.parser')
    q = 0
    a = o.find("strong")
    while a.string.__eq__("Operator") == False:
        a = a.find_next("strong")
        speaker = a.string
    #skip to the talker
    #a = a.find_next("strong")
    #start to talk
    a = a.find_next("p")
    while a.find_next('p') != None:
        if to_check_strong(a) == False:
            if (q == 0):
                output.append((num,speaker,"P",a.string))
                #represent.append(a.string)
            elif q ==1 :
                output.append((num,speaker,"Q",a.string))
                #question.append(a.string)
            elif q ==2 :
                output.append((num,speaker,"A",a.string))
                #answer.append(a.string)
            num +=1
        else:
            speaker = a.string
            if to_check_span(a):
                if to_check_q_a(a):
                    q = 1
                else:
                    q = 2
            else:
                q = 0
        a = a.find_next('p')
    #output = []
    #output.append(represent)
    #output.append(question)
    #output.append(answer)
    #print(output)
    return output

def to_check_strong(node) -> True or False:
    if node.next.name == None:
        return False
    elif node.next.name.__eq__("strong"):
        return True
    return False
def to_check_span(node) -> True or False:
    if node.next.next.name == None:
        return False
    if node.next.next.name.__eq__("span"):
        return True
    return False
def to_check_q_a(node) -> True or False:
    if node.next.next['class'] == None:
        return False
    if node.next.next['class'][0].__eq__("question"):
        return True
    return False

#to store the data
def store(company, document_name, list):
    if not os.listdir("SP500SentenceListAfterClean0402").__contains__(company):
            os.makedirs("SP500SentenceListAfterClean0402/"+company)
    name = "SP500SentenceListAfterClean0402/"+company+"/"+document_name+".json"
    json.dump(list,open(name,"w"))

def main()-> tuple:
    """
    Complete the extraction and analysis from json file using the beautiful soup.
    :return: The transcript data after analysis
    """
    audio = article_with_audio()
    for each in audio:
        try:
            js = take_article_content(each[0],each[1])
            output = beautifulsoup(js)
            store(each[0],each[1],output)
        except:
            print(each[0],each[1])
            continue

#main()
