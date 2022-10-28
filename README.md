Introduction：
------

0.(Most important) What python files could run?!
------
File 1. [Result1_and_2_Models.py](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Result1_and_2_Models.py)

File 2. [Result3_Models.py](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Result3_Model.py)

These two documents could recur all results in the final report.

It could produce all figures in the final report, and all tables (manually) in the final report.

### For the File 1: Result1_and_2_Models.py:
Dataset is complete for the File 1. The file will request you to input 0 or 1 or 2.

Insert 0 to do 50 times data shuffle and show the training data is with high uncertainty, showing by the graph.

Insert 1 to recur the result 1 in the final report (train and back-test the best model and best loss function)

Insert 2 to recur the result 2 in the final report (train and back-test the BNN models with different confidence level)

### For the File 2: Result3_Models.py:
It will show the result 3 in the final report (the NLP models' result)

!!!!Be careful, because the data for the File 2 is so big (6GB), we cannot upload all data, so we only upload a mini data set.


1.All other files:
------
In general, all files are in 3 stages. The stage 1 and 2 are data collection and data cleaning.
The Stage 3 is model building, training, testing, and showing results.

### Stage1 file:
About the companies conference call JSON file and audio file collection and preprocessing.
File 1. [DataCleaning_Audio_feature_extraction.py](DataCleaning_Audio_feature_extraction.py) -> audio feature extraction from audio

File 2. [DataCleaning_Text_Extraction.py](DataCleaning_Text_Extraction.py) -> text extraction from json file

File 3. [DataCleaning_Text_feature_extraction.py](DataCleaning_Text_feature_extraction.py) -> text feature extraction from text

File 4. [DataCollection_audio_collection.py](DataCollection_audio_collection.py) -> collect audio

File 5. [DataCollection_text_collection.py](DataCollection_text_collection.py) -> collect transcript as JSON file.

Why they cannot run?
1. File1: It needs to install the Aeneas and Parsemouth to make it run and the total dataset > 700GB (all audio clips).
2. File2: it involves systematically deal with a large amount of document.
3. File3: It need to upload the Glove300 (10GB too big to upload)
4. File4: It costs money to activate the account to build the paid proxy connection (I do not pay money for it anymore)
5. File5: It costs money to activate the account to build the paid proxy connection (I do not pay money for it anymore)


### Stage2 file:
About the option / stock data collection and preprocessing.
1. [Calculated_Implied_Vol.py](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Calculated_Implied_Vol.py) (helper function only, not need to run)
2. [OptionExtraction.py](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/OptionExtraction.py)

Why it cannot run?
File2: It costs money to activate the account to connect API to get the option data (I do not pay money for it anymore)


### Stage3 file:
About the model building, training, testing, and back test
1. [Result1_and_2_Models.py](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Result1_and_2_Models.py)
2. [Result3_Models.py](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Result3_Model.py)

Both of them could be executed.

2.Sample result for Stage 1 and 2 files:
------
Though Stage 1 and 2 files cannot run, I provide the sample result file to you in the Sample result folder.

You could check each subfolder to see the sample input and the sample output. It could give you some ideas about how data is collected and preprocessed.

*The Output of OptionExtraction.py file (Stage2) is the training data [Volatiloty_data](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Training_Data/train_data.json)

Sample results of Stage 1 files:
[Sample result folder](Sample%20result)

3.Dataset:
------
The dataset for Result1_and_2_Models.py is complete, in the Train_Data folder, call 'train_data.json'
[Volatiloty_data](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Training_Data/train_data.json)

The dataset for Result3_Models.py is incomplete, in the Train_Data folder, call 'train_data_multi_modal_small.json'
[NLP_data](Stage2-3_Option%20and%20Stock%20data%20collection%20and%20model/Training_Data/train_data_multi_modal_small.json)

4.More details:
------
For more rationales and details, you could look at the code comment directly or look at our final report.
