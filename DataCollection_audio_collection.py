import json
import os
count = 0
import DataCollection_text_collection

"""
Stage: Stage 1 file

Document type: Data collection only.

Main purpose: 1. To capture the corresponding audio after we get the transcript.

Need to run? No.

Dependency:
    use -> DataCollection_text_collection.py
    be used -> None

Methods:
    search_audio_path(): to collect the audio path from the dir  

"""

def search_audio_path():
    """
    to collect the audio path from the dir
    """
    count = 0
    check = json.load(open("/Volumes/My Passport/Research Data/SP500Audio/CheckComplete.json"))
    sp500list = os.listdir("JsonSp500/")
    for each in sp500list:
        if os.listdir("/Volumes/My Passport/Research Data/SP500Audio").__contains__(each):
            list = []
            for files in os.listdir("JsonSp500/"+each):
                js = json.load(open("JsonSp500/"+each+"/"+files))
                #print(js.keys())
                if js.__contains__("data"):
                    #print(js['data']['attributes']['transcriptPath'])
                    if js['data']['attributes']['transcriptPath'] != None:
                        list.append(js['data']['attributes']['transcriptPath'])
            for audio in list:
                name = audio[audio.rfind("/")+1:]
                if not os.listdir("/Volumes/My Passport/Research Data/SP500_stopword_semantics/" + each).__contains__(name):
                    print("debug download!")
                    DataCollection_text_collection.try_get_sp500_mp3(DataCollection_text_collection.build_proxy_vpn(), each, audio)
                    count+=1
                    print("debug download!: " + str(count))
        if not os.listdir("/Volumes/My Passport/Research Data/SP500Audio").__contains__(each):
            os.makedirs("/Volumes/My Passport/Research Data/SP500_stopword_semantics/"+each)
            list = []
            for files in os.listdir("JsonSp500/"+each):
                js = json.load(open("JsonSp500/"+each+"/"+files))
                #print(js.keys())
                if js.__contains__("data"):
                    #print(js['data']['attributes']['transcriptPath'])
                    if js['data']['attributes']['transcriptPath'] != None:
                        list.append(js['data']['attributes']['transcriptPath'])
            #check[each] = []
            for audio in list:
                print(each, len(list))
                DataCollection_text_collection.try_get_sp500_mp3(DataCollection_text_collection.build_proxy_vpn(), each, audio)
                count+=1
                #check[each].append(audio[audio.rfind("/")+1:])
                #json.dump(check,open("/Volumes/My Passport/Research Data/SP500_stopword_semantics/CheckComplete.json","w"))
                print("count" + str(count))
            #check[each].append("complete")
            #json.dump(check,open("/Volumes/My Passport/Research Data/SP500_stopword_semantics/CheckComplete.json","w"))
    print(count)



