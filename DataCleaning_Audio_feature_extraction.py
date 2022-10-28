from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import json
from pydub import AudioSegment
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import json
from pydub import AudioSegment
from parselmouth.praat import call
import os
import parselmouth
import gc

"""
Reference: some code is from the tutorial of parselmouth and aeneas
https://www.readbeyond.it/aeneas/docs/index.html
https://parselmouth.readthedocs.io/en/stable/
"""

"""
Stage: Stage 1 file

Document type: Data preprocessing for extracting audio feature only.

Main purpose: 1. To extract the audio feature for each sentence

Need to run? No.

Dependency:
    use -> None
    be used -> None

Methods:
    json_transcript_to_txt(json_path, txt_path): From json file transcript to text only, meeting the requirement
    force_alignment(audio_path,text_path,align_json_output_path): It is to match the audio and the text, producing the json file with transcript, and the time period in the audio
    clip_the_audio(audio_path,json_output_path): Cut the audio based on the output from the force_alignment
    clip_audio_all(): Cut all audios
    processing_for_all_json_to_txt(): Systematically transfer all json file of transcript to text file
    txt_audio_to_align_path(): Systematically to complete the force_alignment
    extract_audio_feature(file_path:str) -> list: This method is to extract the audio feature from the audio, including the pitch/intensity/HNR (mean, variance, max_range) and Jitter/Shimmer feature after PCA
    extract_all_audio_feature(): Systematically complete the extract_audio_feature
"""
def json_transcript_to_txt(json_path, txt_path):
    """
    From json file transcript to text only, meeting the requirement
    :param json_path:
    :param txt_path:
    :return: transcript to text only
    """
    a = json.load(open(json_path,"r"))
    str = a[0][3]
    str += "\n"
    for each in a[1:]:
        if(each[3]!= None):
            str += each[3]
            str += '\n'
    txt = open(txt_path,"w")
    txt.write(str)
    txt.close()
    return

def force_alignment(audio_path,text_path,align_json_output_path):
    """
    Reference: some codes from https://www.readbeyond.it/aeneas/docs/index.html
    It is to match the audio and the text, producing the json file with transcript, and the time period in the audio
    :param audio_path:
    :param text_path:
    :param align_json_output_path: the path of output
    :return:
    """
    # create Task object
    config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = audio_path
    #json_transcript_to_txt()
    task.text_file_path_absolute = text_path
    task.sync_map_file_path_absolute = align_json_output_path

    # process Task
    ExecuteTask(task).execute()

    # output sync map to file
    task.output_sync_map_file()
    del task, config_string
    gc.collect()
#force_alignment()

def clip_the_audio(audio_path,json_output_path):
    """
    Cut the audio based on the output from the force_alignment
    :param audio_path:
    :param json_output_path:
    :return: the audio sequences after cutting.
    """
    os.chdir(".")
    song = AudioSegment.from_mp3(audio_path)
    match = json.load(open(json_output_path,'r'))
    match = match['fragments']
    num = 0
    os.mkdir(audio_path[:audio_path.rfind(".")])
    for each in match:
        clip = song[float(each['begin'])*1000:float(each['end'])*1000]
        path = audio_path[:audio_path.rfind(".")]+audio_path[audio_path.rfind("/"):audio_path.rfind(".")+1]
        clip.export(path+str(num)+".wav",format='wav')
        num+=1
    return

def clip_audio_all(path):
    """
    Cut all audios
    :return: sequences after cutting all audios
    """
    audio_list = []
    json_list = []
    os.chdir(path)
    n = 0
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if n % 100 ==0:
                print(n)
            txt_path = os.path.join(root, name)
            if txt_path[-3:].__eq__("mp3"):
                audio_list.append(txt_path)
                n+=1
            if txt_path[-4:].__eq__("json"):
                json_list.append(txt_path)
                n+=1
    for audio_each in audio_list:
        num = audio_each[audio_each.rfind('/'):audio_each.rfind(".")]
        for txt_each in json_list:
            if txt_each[txt_each.rfind("/"):txt_each.rfind(".")].__eq__(num):
                try:
                    clip_the_audio(audio_each,txt_each)
                except:
                    print(txt_each)
                n+=1
                if n % 100 ==0:
                    print(n)
                continue

#clip_audio_all()

def processing_for_all_json_to_txt(path):
    """
    Systematically transfer all json file of transcript to text file
    :return:
    """
    os.chdir(path)
    n=0
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if n % 100 ==0:
                print(n)
            json_path = os.path.join(root, name)
            if json_path[-4:].__eq__("json"):
                txt_path = json_path[:json_path.rfind(".")+1] + "txt"
                json_transcript_to_txt(json_path,txt_path)
                n+=1
#processing_for_all_json_to_txt()

def txt_audio_to_align_path(path):
    """
    Systematically to complete the force_alignment
    :return:
    """
    os.chdir(path)
    n=0
    txt_list = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if n % 100 ==0:
                print(n)
            txt_path = os.path.join(root, name)
            if txt_path[-3:].__eq__("txt"):
                txt_list.append(txt_path)
                n+=1
    audio_list = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if n % 100 ==0:
                print(n)
            txt_path = os.path.join(root, name)
            if txt_path[-3:].__eq__("mp3"):
                audio_list.append(txt_path)
                n+=1
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if n % 100 ==0:
                print(n)
            txt_path = os.path.join(root, name)
            if txt_path[-4:].__eq__("json"):
                audio_list.remove(txt_path[:-4]+'mp3')
    print(len(audio_list))

    for audio_each in audio_list:
        if n % 100 ==0:
            print(n)
        num = audio_each[audio_each.rfind('/'):audio_each.rfind(".")]
        for txt_each in txt_list:
            if txt_each[txt_each.rfind("/"):txt_each.rfind(".")].__eq__(num):
                json_name = audio_each[:audio_each.rfind('.')+1] + "json"
                txt_each = path + txt_each[1:]
                try:
                    force_alignment(audio_each,txt_each,json_name)
                    n+=1
                except:
                    print(txt_each);
                gc.collect()
                continue;
#txt_audio_to_align_path()
def extract_audio_feature(file_path:str) -> list:
    '''
    This method is to extract the audio feature from the audio,
    including the pitch/intensity/HNR (mean, variance, max_range) and Jitter/Shimmer feature after PCA
    Reference: https://parselmouth.readthedocs.io/en/stable/
    :param file_path: The audio path
    :return: should return a list with 3 * 3 + 5 + 6 = 20 element list
    '''
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    import json
    from pydub import AudioSegment
    from parselmouth.praat import call
    import parselmouth

    sound = parselmouth.Sound(file_path)

    #To get the pitch
    pitch = sound.to_pitch()
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    standard_deviation_pitch = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    range = max_pitch - min_pitch

    #To get the intensity
    intensity = sound.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0, 0)
    min_intensity = call(intensity, "Get minimum", 0, 0,"Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0,"Parabolic")
    sd_intensity = call(intensity, "Get standard deviation", 0, 0)
    range_intensity = max_intensity - min_intensity

    #To get the harmonicity
    harmonicity = sound.to_harmonicity()
    mean_harmonicity  = call(harmonicity , "Get mean", 0, 0)
    min_harmonicity  = call(harmonicity , "Get minimum", 0, 0,"Parabolic")
    max_harmonicity  = call(harmonicity , "Get maximum", 0, 0,"Parabolic")
    sd_harmonicity  = call(harmonicity , "Get standard deviation", 0, 0)
    range_harmonicity = max_harmonicity - min_harmonicity

    #To get the Jitter and shimmer
    #Reference: The code from line 99 - 110 is from https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 70, 400) # The 70 and 400 here are the hyper-parameters to show the min-f0 and max-f0
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    outputlist = [mean_pitch,standard_deviation_pitch,range,mean_intensity, sd_intensity, range_intensity,
                  mean_harmonicity,sd_harmonicity,range_harmonicity,
                  localJitter,localabsoluteJitter,rapJitter,ppq5Jitter,ddpJitter,
                  localShimmer,localdbShimmer,apq3Shimmer,aqpq5Shimmer, apq11Shimmer,ddaShimmer]
    #print(outputlist)
    return outputlist

def extract_all_audio_feature(path):
    """
    Systematically complete the extract_audio_feature
    :return:
    """
    "The complete list"
    complete_list = json.load(open(path+ "Complete_list.txt","r"))
    print(len(complete_list))
    trouble_list = json.load(open(path + "Trouble_list.txt","r"))

    "Txt_list is the list of each json file's path"
    os.chdir(path + "/SP500SentenceListAfterClean0402")
    n=0
    txt_list = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:

            txt_path = os.path.join(root, name)
            if txt_path[-4:].__eq__("json"):
                txt_list.append(txt_path)

    "Each_txt = each json file path"
    for each_txt in txt_list:
        os.chdir(path + "/SP500SentenceListAfterClean0402")
        json_file = json.load(open(each_txt,"r"))

        "Name = unique stock_index/number"
        name = each_txt[2:-5]
        print(name)
        if name[name.find("/")+1:] not in complete_list and name[name.find("/")+1:] not in trouble_list:
            try:
                os.chdir(path + name)
                for root, dirs, files in os.walk(".", topdown=False):
                    for names in files:
                        txt_path = os.path.join(root, names)
                        num = txt_path[:-4]
                        num = num[num.rfind(".")+1:]

                        "Audio paramters"
                        list = extract_audio_feature(txt_path)

                        "num ensures the match"
                        if int(num) == json_file[int(num)][0]:
                            json_file[int(num)].append(list)
                            n+=1
                            print(n)
                        else:
                            print(name + "    fail")
                            trouble_list.append(name[name.find("/")+1:])
                            trouble_file = open(path + "/Trouble_list.txt","w")
                            trouble_file.write(json.dumps(trouble_list))
                            trouble_file.close()
                            continue
                complete_list.append(name[name.find("/")+1:])
                complete_file = open(path + "/Complete_list.txt","w")
                complete_file.write(json.dumps(complete_list))
                complete_file.close()
                os.chdir(path + "SP500SentenceListAfterClean0402")
                json_file_open = open(each_txt,"w")
                json_file_open.write(json.dumps(json_file))
                json_file_open.close()

            except:
                trouble_list.append(name[name.find("/")+1:])
                trouble_file = open(path + "/Trouble_list.txt","w")
                trouble_file.write(json.dumps(trouble_list))
                trouble_file.close()


#extract_all_audio_feature()
