import numpy as np
import tensorflow
from keras.layers import Input, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional,Concatenate
from keras.models import Model
from keras import backend as K
import json
import os
import pandas
from attention import Attention
np.random.seed(2)  # for reproducibility
import json
import random
import matplotlib.pyplot as plt
import math
"""
Stage: Stage 3 file
Document type: NLP Models for result 3.

Need to run? YES!!!!

Main purpose: To build our model and get the result

Dependency:
    use -> None
    be used -> None

Methods:
    Part 1: Loss function
        Sortino_Ratio_as_loss_fn: Sortino ratio calculation as loss function
    
    Part 2: Back-test
        Backtest: Normal back-test
    
    Part 3: Data preprocessing
        Data_Preprocessing: to produce the training set and test set; input and labels; split to semantic, embedding and vocal features
        
    Part 4: Models
        Model_building: to build the NLP model
    
"""
random.seed(2)
tensorflow.test.gpu_device_name()

print("Tensorflow version " + tensorflow.__version__)


gpus = tensorflow.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    gpu0 = gpus[0]  # setting GPU-0
    tensorflow.config.experimental.set_memory_growth(gpu0, True)
    tensorflow.config.set_visible_devices([gpu0], "GPU")


"""
Group of loss function
"""
def Sortino_Ratio_as_loss_fn(y_true, y_pred):
    """
    The idea is similar, also using the backtest.
    However, we are more focusing on the Sortino Ratio as the strategy effectiveness.

    Calculation of Sortino Ratio = (Return_rate - risk-free return rate) / Downward-risk

    *We here to assume the risk-free rate = 0

    @output : Sortino_Ratio as scalar
    """
    #call_position = tensorflow.ones(y_true.shape[0])
    put_position = tensorflow.abs(y_true[:, 4]) / tensorflow.abs(y_true[:, 12])  # Position amount = Call Delta / Put Delta as hedging
    call_earn = tensorflow.cast((y_true[:, 3] - y_true[:, 2]), dtype=tensorflow.float32)
    put_earn = tensorflow.cast(put_position * (y_true[:, 11] - y_true[:, 10]), dtype=tensorflow.float32)
    amount_money = tensorflow.cast(y_true[:, 2], dtype=tensorflow.float32) + tensorflow.cast(put_position * y_true[:,10], dtype=tensorflow.float32)
    result = tensorflow.cast(y_pred[:, 0], dtype=tensorflow.float32) * ((call_earn + put_earn) / amount_money)


    result_mask = tensorflow.cast(tensorflow.boolean_mask(result,result<=0),dtype=tensorflow.float32)

    std = tensorflow.math.reduce_std(result_mask)
    if std < 0.1:
        std = tensorflow.constant([0.1],dtype=tensorflow.float32)
    elif result_mask.shape[0] == 0:
        std = tensorflow.constant([0.1],dtype=tensorflow.float32)
    elif tensorflow.math.is_nan(std):
        std = tensorflow.constant([0.1],dtype=tensorflow.float32)
    sortino_ratio = (tensorflow.reduce_mean(result)) / (std) * math.sqrt(251)

    return - sortino_ratio  # To minimize the loss function

def Backtest(y_true, y_pred,name):
    """
    y_true as input:
        0. Position size
        1. Predicted call Implied Volatility
        2. Predicted put Implied Volatility
    Calculation is the profit based on the backtest only (Real profit).

    Profit approx = position_amount % * (call position * call_price_change) + (put position * put_price_change) / (total amount of sell/buy options) * 100

    """
    def Sharpe_Ratio(y_true, y_pred):
        """
        The idea is similar, also using the backtest.
        However, we are more focusing on the Sharpe Ratio as the strategy effectiveness.

        Calculation of Shapre Ratio = (Return_rate - risk-free return rate) / Total risk

        *We here to assume the risk-free rate = 0

        @output : Sharpe_Ratio as scalar
        """
        call_position = tensorflow.ones(y_true.shape[0])
        put_position = tensorflow.abs(y_true[:, 4]) / tensorflow.abs(
            y_true[:, 12])  # Position amount = Call Delta / Put Delta as hedging
        call_earn = tensorflow.cast(call_position * (y_true[:, 3] - y_true[:, 2]), dtype=tensorflow.float64)
        put_earn = tensorflow.cast(put_position * (y_true[:, 11] - y_true[:, 10]), dtype=tensorflow.float64)
        amount_money = tensorflow.cast(call_position * y_true[:, 2], dtype=tensorflow.float64) + tensorflow.cast(put_position * y_true[:,10], dtype=tensorflow.float64)
        result = tensorflow.cast(y_pred[:, 0], dtype=tensorflow.float64) * ((call_earn + put_earn) / amount_money)

        """
        earn_list = []
        earn = 1
    
        for each in result:
            earnnew = 0.1 * earn * (1 - each)
            earn = earn * 0.9 + earnnew
            
            
                earn_list.append(earn - 1)
                earn = 1
        """

        sharpe_ratio = tensorflow.reduce_mean(result) / tensorflow.math.reduce_std(result) * math.sqrt(251)
        print('earn: ', tensorflow.reduce_mean(result))
        print('std: ', tensorflow.math.reduce_std(result))
        print("sharpe_ratio: ", str(sharpe_ratio))
        return - sharpe_ratio  # To minimize the loss function

    def Sortino_Ratio_as_loss_fn(y_true, y_pred):
        """
        The idea is similar, also using the backtest.
        However, we are more focusing on the Sortino Ratio as the strategy effectiveness.

        Calculation of Sortino Ratio = (Return_rate - risk-free return rate) / Downward-risk

        *We here to assume the risk-free rate = 0

        @output : Sortino_Ratio as scalar
        """

        #call_position = tensorflow.ones(y_true.shape[0])
        put_position = tensorflow.abs(y_true[:, 4]) / tensorflow.abs(y_true[:, 12])  # Position amount = Call Delta / Put Delta as hedging
        call_earn = tensorflow.cast((y_true[:, 3] - y_true[:, 2]), dtype=tensorflow.float32)
        put_earn = tensorflow.cast(put_position * (y_true[:, 11] - y_true[:, 10]), dtype=tensorflow.float32)
        amount_money = tensorflow.cast(y_true[:, 2], dtype=tensorflow.float32) + tensorflow.cast(put_position * y_true[:,10], dtype=tensorflow.float32)
        result = tensorflow.cast(y_pred[:, 0], dtype=tensorflow.float32) * ((call_earn + put_earn) / amount_money)

        mask = result < 0
        result_mask = tensorflow.boolean_mask(result,mask)

        #mask_mean = result != 0
        #result_mean = tensorflow.boolean_mask(result,mask_mean)

        sortino_ratio = tensorflow.reduce_mean(result) / tensorflow.math.reduce_std(result_mask) * math.sqrt(251)
        print("sortino ratio: ", sortino_ratio)
        return - sortino_ratio  # To minimize the loss function
    # y_true = tensorflow.cast(y_true, dtype=tensorflow.float64)
    # y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float64)
    y_true_copy = y_true
    y_true = y_true[:,:16]
    y_true = y_true.astype(np.float64)
    y_pred = tensorflow.cast(y_pred,dtype=tensorflow.float64)
    call_position = tensorflow.ones(y_true.shape[0])
    put_position = tensorflow.abs(y_true[:, 4]) / tensorflow.abs(y_true[:, 12])  # Position amount = Call Delta / Put Delta as hedging
    call_earn = tensorflow.cast(call_position * (y_true[:, 3] - y_true[:, 2]), dtype=tensorflow.float64)
    put_earn = put_position * (y_true[:, 11] - y_true[:, 10])
    amount_money = tensorflow.cast(call_position * y_true[:, 2], dtype=tensorflow.float64) + put_position * y_true[:,10]
    result = - y_pred[:, 0] * ((call_earn + put_earn) / amount_money) + 0.005  # Assume 0.5% trading fee

    earn = 1
    n = 0
    win = 0
    lose = 0
    buy = 0
    sell = 0
    no_action = 0
    """
    Check the withdraw
    """
    withdraw = 0
    withdraw_at_peak = 1
    withdraw_at_bottom = 1
    max_withdraw = 0
    max_withdraw_peak = 1
    max_withdraw_bottom = 1
    earn_history = [1]
    for each in result:
        # To update the peak if it is higher.
        if earn >= withdraw_at_peak:
            withdraw_at_peak = earn
            withdraw_at_bottom = earn
        elif earn <= withdraw_at_bottom:
            withdraw_at_bottom = earn
            withdraw = 1 - withdraw_at_bottom / withdraw_at_peak
            if withdraw > max_withdraw:
                max_withdraw = withdraw
                max_withdraw_peak = withdraw_at_peak
                max_withdraw_bottom = withdraw_at_bottom
        """
        End of check of withdraw
        """
        earnnew = 0.1 * earn * (1 - each)
        earn = earn * 0.9 + earnnew
        earn_history.append(earn)
        #print(y_pred[n], -result[n], y_true[n, 0:2], y_true[n, 8:10],y_true_copy[n,16:18])
        if y_pred[n,0] > 0:
            buy += 1
        elif y_pred[n,0] < 0:
            sell += 1
        elif y_pred[n,0] == 0:
            no_action +=1
        n += 1
        #print(earn)
        if each > 0:
            lose += 1
        elif each < 0:
            win += 1

    print('win: ', win, "lose: ", lose)
    print('buy: ', buy, "sell: ", sell, 'no action: ', no_action)
    print('total earn: ', earn)
    print('withdraw: ', max_withdraw, max_withdraw_peak, max_withdraw_bottom)
    Sharpe_Ratio(y_true,y_pred)
    Sortino_Ratio_as_loss_fn(y_true,y_pred)
    plot_data[name] = earn_history
    return - earn

def Make_plot():
    """
    Print the result
    :return:
    """
    for each in plot_data.keys():
        plt.plot(list(range(len(plot_data[each]))),plot_data[each],label=each)

    plt.title('The Trading result graph')
    plt.xlabel('times')
    plt.ylabel('total value')
    plt.legend()
    plt.savefig('foo.png', dpi=100, bbox_inches='tight')
    plt.show()

def Data_Preprocessing():
    """
    This method is to pre-process the data to create 4 groups:
        1. Train data as np.arrays = [
            1. Price change on the open price
            2. Volatility of the past month
            3. Implied-Volatility for open price of call option
            4. Implied_Volatility for close price of put option
            ]

        2. Train label as np.arrays = [
            1-2 Open/Close call implied volatility
            3-4 Open/Close call price
            5-8 Delta, Gamma, Vega, Rho

            9-10 Open/Close put implied volatility
            11-12 Open/Close put price
            13-16 Delta, Gamma, Vega, Rho
        3. Test data as np.arrays
        4. Test label as np.arrays
    """
    dataset = json.load(open("Training_Data/train_data_multi_modal_small.json",'r'))
    random.shuffle(dataset)
    dataset_np = np.array(dataset, dtype=object)

    train_data_before = dataset_np[:, 0].tolist()
    basic = []
    sem = []
    emb = []
    vol = []
    for each_sample in train_data_before:
        basic.append(each_sample[:4])
        sem.append(each_sample[4])
        emb.append(each_sample[5])
        vol.append(each_sample[6])
    dataset_basic = np.array(basic)
    dataset_sem = np.array(sem)
    dataset_emb = np.array(emb)
    dataset_vol = np.array(vol)

    """
    normalization of vol
    """
    def normalization(vector):
        output = vector
        m = np.nanmean(vector,axis = (1))
        s = np.nanstd(vector,axis = (1))
        for each in range(vector.shape[0]):
          output[each] = (vector[each] - m[each]) / s[each]
        return output

    #dataset_sem = (dataset_sem - np.mean(dataset_sem, axis=(0,1))) / np.std(dataset_sem,axis=(0,1))
    #dataset_emb = (dataset_emb - np.mean(dataset_emb, axis=(0,1))) / np.std(dataset_emb,axis=(0,1))
    dataset_vol = normalization(dataset_vol)


    nan = np.isnan(dataset_vol)
    dataset_vol[nan] = -1

    dataset_np_label = np.array(dataset_np[:, 1].tolist())

    output_dataset_basic_data = dataset_basic[:int(len(dataset_np_label) * 0.8)]
    output_dataset_sem_data = dataset_sem[:int(len(dataset_np_label) * 0.8)]
    output_dataset_emb_data = dataset_emb[:int(len(dataset_np_label) * 0.8)]
    output_dataset_vol_data = dataset_vol[:int(len(dataset_np_label) * 0.8)]
    output_dataset_basic_label = dataset_basic[int(len(dataset_np_label) * 0.8):]
    output_dataset_sem_label = dataset_sem[int(len(dataset_np_label) * 0.8):]
    output_dataset_emb_label = dataset_emb[int(len(dataset_np_label) * 0.8):]
    output_dataset_vol_label = dataset_vol[int(len(dataset_np_label) * 0.8):]


    output_train_label = dataset_np_label[:int(len(dataset_np_label) * 0.8)]
    output_test_label = dataset_np_label[int(len(dataset_np_label) * 0.8):]

    return [output_dataset_basic_data, output_dataset_sem_data, output_dataset_emb_data, output_dataset_vol_data], output_train_label, [output_dataset_basic_label, output_dataset_sem_label, output_dataset_emb_label, output_dataset_vol_label], output_test_label

def Model_building(train_data, train_label, test_data, test_label):
    """
    NLP model building
    """
    train_data_basic = tensorflow.constant(train_data[0],dtype=tensorflow.float32)
    train_data_sem = tensorflow.constant(train_data[1],dtype=tensorflow.float32)
    train_data_emb = tensorflow.constant(train_data[2],dtype=tensorflow.float32)
    train_data_vol = tensorflow.constant(train_data[3],dtype=tensorflow.float32)

    test_data_basic = tensorflow.constant(test_data[0],dtype=tensorflow.float32)
    test_data_sem = tensorflow.constant(test_data[1],dtype=tensorflow.float32)
    test_data_emb = tensorflow.constant(test_data[2],dtype=tensorflow.float32)
    test_data_vol = tensorflow.constant(test_data[3],dtype=tensorflow.float32)

    train_label = tensorflow.constant(train_label[:,:16],dtype=tensorflow.float32)
    def Easy_Model(model_save_path,train_data, train_label, test_data, test_label):
        """
        This model with input:
            1. Price change on the open price
            2. Volatility of the past month
            3. Implied-Volatility for open price of call option
            4. Implied_Volatility for close price of put option

        This model with label:
            1. Train data as np.arrays = [
                1. Price change on the open price
                2. Volatility of the past month
                3. Implied-Volatility for open price of call option
                4. Implied_Volatility for close price of put option
                ]

            2. Train label as np.arrays = [
                1-2 Open/Close call implied volatility
                3-4 Open/Close call price
                5-8 Delta, Gamma, Vega, Rho

                9-10 Open/Close put implied volatility
                11-12 Open/Close put price
                13-16 Delta, Gamma, Vega, Rho
        """
        #train_data, train_label, test_data, test_label = Data_Preprocessing()
        #train_data = tensorflow.constant(train_data,dtype=tensorflow.float32)
        #train_label = tensorflow.constant(train_label[:,:16],dtype=tensorflow.float32)

        inputs = Input(shape=(4,))
        benchmark_fc = Dense(3, activation='relu')(inputs)
        final_output = Dense(1, activation='tanh')(benchmark_fc)

        model = Model(inputs, final_output)
        """
        Hyperparameters: Loss function, Sample_weight_mode, batch_size, epochs...
        """
        # creating the model in the TPUStrategy scope means we will train the model on the TPU
        model.compile(optimizer='adam', loss=Sortino_Ratio_as_loss_fn, run_eagerly=False)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(train_data, train_label,
                  epochs=200,
                  batch_size=20,
                  # sample_weight=None,#initial_weight,
                  shuffle=True)
            # callbacks=[early_stopping]),
            # validation_split=0.2)
        Backtest(test_label, model.predict(test_data),'EasyModel')
        return model

    """
    The Input shape is to tell (a,b) that a is the number of sentences, b is the dimensions of each sentence
    """
    model = Easy_Model('bc-LSTM/Multimodal_easy_model.tf',train_data_basic,train_label,test_data_basic,test_label)
    benchmark_predict = model.predict(train_data_basic)
    modal_data_semantics = Input(shape=(400,8))
    modal_data_embeddings = Input(shape=(400,300))
    modal_data_vocal = Input(shape=(400,20))
    """
    The use of the masking is to tell not calculating any rows with -1 only 
    """
    masked_semantics = Masking(mask_value=-1)(modal_data_semantics)
    masked_embeddings = Masking(mask_value=-1)(modal_data_embeddings)
    masked_vocal = Masking(mask_value=-1)(modal_data_vocal)

    """
    LSTM(number) number means the layer into the Dense Layer
    """
    lstm_semantics = Bidirectional(LSTM(8, activation='tanh', return_sequences=True, dropout=0.6))(masked_semantics)
    lstm_embeddings = Bidirectional(LSTM(300, activation='tanh', return_sequences=True, dropout=0.6))(masked_embeddings)
    lstm_vocal = Bidirectional(LSTM(20, activation='tanh', return_sequences=True, dropout=0.6))(masked_vocal)

    """
    Dropout rate (d) to avoid the overfitting
    """
    inter_semantics = Dropout(0.9)(lstm_semantics)
    inter_embeddings = Dropout(0.9)(lstm_embeddings)
    inter_vocal = Dropout(0.9)(lstm_vocal)

    """
    Final Output (e), e represent the output dimensions
    """
    output_semantics = TimeDistributed(Dense(8, activation='relu'))(inter_semantics)
    output_embeddings = TimeDistributed(Dense(100, activation='relu'))(inter_embeddings)
    output_vocal = TimeDistributed(Dense(10, activation='relu'))(inter_vocal)

    output2_semantics = Attention(units=8)(output_semantics)
    output2_embeddings = Attention(units=100)(output_embeddings)
    output2_vocal = Attention(units=10)(output_vocal)

    #These three model is used for building the multi_modal_model_data
    sem_uni_model = Model(modal_data_semantics,output_semantics)
    emb_uni_model = Model(modal_data_embeddings,output_embeddings)
    vol_uni_model = Model(modal_data_vocal,output_vocal)

    final_output_semantics = Dense(1, activation='tanh')(output2_semantics)
    final_output_embeddings = Dense(1, activation='tanh')(output2_embeddings)
    final_output_vocal = Dense(1, activation='tanh')(output2_vocal)

    """
    We use a Concatenate layer here to combine benchmark and modal data
    """
    benchmark_sem = Input(shape=(1,))
    benchmark_emb = Input(shape=(1,))
    benchmark_vol = Input(shape=(1,))

    combined_data_semantics = Concatenate()([benchmark_sem,final_output_semantics])
    combined_data_embeddings = Concatenate()([benchmark_emb,final_output_embeddings])
    combined_data_vocal = Concatenate()([benchmark_vol,final_output_vocal])

    trade_decision_semantics = Dense(1, activation='tanh')(combined_data_semantics)
    trade_decision_embeddings = Dense(1, activation='tanh')(combined_data_embeddings)
    trade_decision_vocal = Dense(1, activation='tanh')(combined_data_vocal)

    unimodel_semantics = Model([benchmark_sem,modal_data_semantics],trade_decision_semantics)
    unimodel_embeddings = Model([benchmark_emb,modal_data_embeddings],trade_decision_embeddings)
    unimodel_vocal = Model([benchmark_vol,modal_data_vocal],trade_decision_vocal)
    unimodel_semantics.compile(optimizer='adam', loss=Sortino_Ratio_as_loss_fn)
    unimodel_embeddings.compile(optimizer='adam', loss=Sortino_Ratio_as_loss_fn)
    unimodel_vocal.compile(optimizer='adam', loss=Sortino_Ratio_as_loss_fn)


    benchmark_predict_test = model.predict(test_data_basic)
    unimodel_semantics.fit([benchmark_predict,train_data_sem], train_label,
              epochs=200,
              batch_size=20,
              #sample_weight=None,#initial_weight,
              shuffle=True),
              #callbacks=[early_stopping]),
              #validation_split=0.2)
    print('unimodal_semantics_backtest')
    Backtest(test_label,unimodel_semantics.predict([benchmark_predict_test,test_data_sem]),'Uni_semantic')

    unimodel_embeddings.fit([benchmark_predict,train_data_emb], train_label,
              epochs=200,
              batch_size=20,
              #sample_weight=None,#initial_weight,
              shuffle=True),
              #callbacks=[early_stopping]),
              #validation_split=0.2)
    print('unimodal_embeddings_backtest')
    Backtest(test_label,unimodel_embeddings.predict([benchmark_predict_test,test_data_emb]),'Uni_embedding')

    unimodel_vocal.fit([benchmark_predict,train_data_vol], train_label,
              epochs=200,
              batch_size=20,
              #sample_weight=None,#initial_weight,
              shuffle=True),
              #callbacks=[early_stopping]),
              #validation_split=0.2)
    print('unimodal_vocal_backtest')
    Backtest(test_label,unimodel_vocal.predict([benchmark_predict_test,test_data_vol]),'Uni_vocal')
    #Now complete the Unimodal training, and turning to the multi-modal training

    sem = sem_uni_model.predict(train_data_sem)
    emb = emb_uni_model.predict(train_data_emb)
    vol = vol_uni_model.predict(train_data_vol)
    concat = Concatenate()([sem,emb,vol])

    '''
    The multi-modal model building; actually doing the same thing again but just using concat vector
    '''

    input = Input(shape = (400,118))
    masked = Masking(mask_value=-1)(input)
    lstm = Bidirectional(LSTM(200, activation='tanh', return_sequences=True, dropout=0.6))(masked)
    inter = Dropout(0.9)(lstm)
    output = TimeDistributed(Dense(200, activation='relu'))(inter)
    output2 = Attention(units=200)(output)
    final_output = Dense(1, activation='tanh')(output2)
    benchmark = Input(shape=(1,))

    combined_data = Concatenate()([benchmark,final_output])
    trade_decision = Dense(1, activation='tanh')(combined_data)
    multimodal_model = Model([benchmark,input],trade_decision)
    multimodal_model.compile(optimizer='adam', loss=Sortino_Ratio_as_loss_fn)
    multimodal_model.fit([benchmark_predict,concat], train_label,
              epochs=200,
              batch_size=20,
              #sample_weight=None,#initial_weight,
              shuffle=True),
              #callbacks=[early_stopping]),
              #validation_split=0.2)
    print('multimodal_backtest')

    sem_test = sem_uni_model.predict(test_data_sem)
    emb_test = emb_uni_model.predict(test_data_emb)
    vol_test = vol_uni_model.predict(test_data_vol)
    concat_test = Concatenate()([sem_test,emb_test,vol_test])
    Backtest(test_label,multimodal_model.predict([benchmark_predict_test,concat_test]),'Multi_modal')

plot_data = {}
train_data, train_label, test_data, test_label = Data_Preprocessing()
Model_building(train_data, train_label, test_data, test_label)
Make_plot()

