import numpy as np
import tensorflow
from keras.layers import Input,  Dense
from keras.models import Model
import keras
import matplotlib.pyplot as plt

import json
import random
import math
import tensorflow_probability as tfp
random.seed(22)  # for reproducibility
"""
Stage: Stage 3 file
Document type: Models for result 1 and 2 and data shuffle check.

Need to run? YES!!!!

Main purpose: To build our model and get the result

Dependency:
    use -> None
    be used -> None

Methods:
    Part 1: Loss function
        Sharpe_Ratio_as_loss_fn : Sharpe ratio calculation as loss function
        Sortino_Ratio_as_loss_fn: Sortino ratio calculation as loss function
        Volatility_only_expected_profit_loss_fn: profit for average single trade as loss function
    
    Part 2: Back-test
        Backtest: Normal back-test
        Backtest_Bayesian: Back-test for BNN model with Monte-Carlo method
    
    Part 3: Data preprocessing
        Data_Preprocessing: to produce the training set and test set; input and labels
        
    Part 4: Models
        Easy_model: Traditional NN model
        Bayesian_model: BNN model
    
    Part 5: run test
        data_shuffle: run check data shuffle
        result_1: run result 1
        result_2: run result 2
"""
"""
The following is the loss funciton
"""

def Sharpe_Ratio_as_loss_fn(y_true, y_pred):
    """
    The idea is similar, also using the backtest.
    However, we are more focusing on the Sharpe Ratio as the strategy effectiveness.

    Calculation of Shapre Ratio = (Return_rate - risk-free return rate) / Total risk

    *We here to assume the risk-free rate = 0

    @output : Sharpe_Ratio as scalar
    """
    #call_position = tensorflow.ones(y_true.shape[0])
    put_position = tensorflow.abs(y_true[:, 4]) / tensorflow.abs(y_true[:, 12])  # Position amount = Call Delta / Put Delta as hedging
    call_earn = tensorflow.cast((y_true[:, 3] - y_true[:, 2]), dtype=tensorflow.float32)
    put_earn = tensorflow.cast(put_position * (y_true[:, 11] - y_true[:, 10]), dtype=tensorflow.float32)
    amount_money = tensorflow.cast(y_true[:, 2], dtype=tensorflow.float32) + tensorflow.cast(put_position * y_true[:,10], dtype=tensorflow.float32)
    result = tensorflow.cast(y_pred[:, 0], dtype=tensorflow.float32) * ((call_earn + put_earn) / amount_money)
    std = tensorflow.math.reduce_std(result)
    if std < 0.1:
        std = tensorflow.constant([0.1],dtype=tensorflow.float32)
    sharpe_ratio = tensorflow.reduce_mean(result) / std * math.sqrt(251)
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

def Volatility_only_expected_profit_loss_fn(y_true, y_pred):
    """
    y_true as input:
        0. Position size
        1. Predicted call Implied Volatility
        2. Predicted put Implied Volatility
    Calculation is the profit based on the implied Volatility only (Expected profit).

    Profit approx = position_amount % * (call position * vega_call * changeOfImp) + (put position * vega_put * changeOfImp) / (total amount of sell/buy options) * 100

    """
    sharpe = Sharpe_Ratio_as_loss_fn(y_true, y_pred)
    if sharpe > -3 :
        return sharpe + 10
    # sess = tensorflow.Session()
    # with sess.as_default():
    #    y_true_np = y_true.eval(session=sess)
    #    y_pred_np = y_pred.eval(session=sess)
    y_true_np = y_true
    y_pred_np = y_pred
    """
    We assume call position is always 1, so we do not need to use it 
    """
    #call_position = tensorflow.ones(y_true_np.shape[0])
    put_position = tensorflow.abs(y_true_np[:, 4]) / tensorflow.abs(y_true_np[:, 12])  # Position amount = Call Delta / Put Delta as hedging
    all_cost = (y_true_np[:, 2] + put_position * y_true_np[:,10]) * 100
    earning = (y_true_np[:, 6] * (y_true_np[:, 1] - y_true_np[:, 0]) * 100 + put_position * y_true_np[:, 14] * (y_true_np[:, 9] - y_true_np[:, 8]) * 100)


    return tensorflow.reduce_mean(- y_pred_np[:, 0] * earning / all_cost * 100) # + tensorflow.abs(y_pred_np[:, 0]) * 0.5)


"""
The end of the loss function
"""

"""
Backtest model
"""


def Backtest(y_true, y_pred):
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

        mask_mean = result != 0
        result_mean = tensorflow.boolean_mask(result,mask_mean)

        sortino_ratio = tensorflow.reduce_mean(result_mean) / tensorflow.math.reduce_std(result_mask) * math.sqrt(251)
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
    result = - y_pred[:, 0] * ((call_earn + put_earn) / amount_money - 0.005)  # Assume 0.5% trading fee

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
    sharpe = Sharpe_Ratio(y_true,y_pred)
    sortino = Sortino_Ratio_as_loss_fn(y_true,y_pred)
    global num
    plot_data[str(loss[loss_num]) + 'Easy_model: '+ str(num)] = earn_history
    num+=1
    SHARPE.append(sharpe)
    SORTINO.append(sortino)
    TOTAL_EARN.append(earn)
    WITHDRAW.append(max_withdraw)
    return - earn


def Backtest_Bayesian(y_true, y_pred, model, test_data, decision_rate):
    """
    y_true as input:
        0. Position size
        1. Predicted call Implied Volatility
        2. Predicted put Implied Volatility
    Calculation is the profit based on the backtest only (Real profit).

    Profit approx = position_amount % * (call position * call_price_change) + (put position * put_price_change) / (total amount of sell/buy options) * 100

    """
    def Backtest_Bay_Simple(y_true, y_pred):
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
            result = tensorflow.boolean_mask(result, result != 0)

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
            result = tensorflow.boolean_mask(result, result != 0)
            mask = result < 0
            result_mask = tensorflow.boolean_mask(result,mask)

            mask_mean = result != 0
            result_mean = tensorflow.boolean_mask(result,mask_mean)

            sortino_ratio = tensorflow.reduce_mean(result_mean) / tensorflow.math.reduce_std(result_mask) * math.sqrt(251)
            print("sortino ratio: ", sortino_ratio)
            return - sortino_ratio  # To minimize the loss function
        # y_true = tensorflow.cast(y_true, dtype=tensorflow.float64)
        # y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float64)

        y_true = y_true[:,:16]
        y_true = y_true.astype(np.float64)
        y_pred = tensorflow.cast(y_pred,dtype=tensorflow.float64)
        call_position = tensorflow.ones(y_true.shape[0])
        put_position = tensorflow.abs(y_true[:, 4]) / tensorflow.abs(y_true[:, 12])  # Position amount = Call Delta / Put Delta as hedging
        call_earn = tensorflow.cast(call_position * (y_true[:, 3] - y_true[:, 2]), dtype=tensorflow.float64)
        put_earn = put_position * (y_true[:, 11] - y_true[:, 10])
        amount_money = tensorflow.cast(call_position * y_true[:, 2], dtype=tensorflow.float64) + put_position * y_true[:,10]
        result =  y_pred[:, 0] * ((call_earn + put_earn) / amount_money) - tensorflow.abs(y_pred[:, 0]) * 0.005  # Assume 0.5% trading fee
        result = tensorflow.boolean_mask(result, result != 0)

        earn = 1
        n = 0
        win = 0
        lose = 0
        buy = 0
        sell = 0
        no_action = 0
        par = 0
        '''
        Check the buy and sell
        '''
        while n < y_pred.shape[0]:
            if y_pred[n,0] > 0:
                buy += 1
            elif y_pred[n,0] < 0:
                sell += 1
            elif y_pred[n,0] == 0:
                no_action +=1

            n += 1
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
            earnnew = 0.1 * earn * (1 + each)
            earn = earn * 0.9 + earnnew
            earn_history.append(earn)
            #print(y_pred[n], -result[n], y_true[n, 0:2], y_true[n, 8:10],y_true_copy[n,16:18])
            #print(earn)
            if each < 0:
                lose += 1
            elif each > 0:
                win += 1
            elif each == 0:
                par += 1

        print('win: ', win, "lose: ", lose, "par: ", par)
        print('buy: ', buy, "sell: ", sell, 'no action: ', no_action)
        print('total earn: ', earn)
        print('withdraw: ', max_withdraw, max_withdraw_peak, max_withdraw_bottom)
        Sharpe_Ratio(y_true,y_pred)
        Sortino_Ratio_as_loss_fn(y_true,y_pred)
        lebel = str(loss[loss_num]) + " Bayesian Selection "+str(hyperpara_bayesian_decision)+" std"
        plot_data[lebel] = earn_history
        return - earn

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

        sharpe_ratio = tensorflow.reduce_mean(result) / tensorflow.math.reduce_std(result) * math.sqrt(251)
        print('earn: ', tensorflow.reduce_mean(result))
        print('std: ', tensorflow.math.reduce_std(result))
        print("sharpe_ratio: ", str(sharpe_ratio))
        return - sharpe_ratio  # To minimize the loss function
    # y_true = tensorflow.cast(y_true, dtype=tensorflow.float64)
    # y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float64)
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


        mask_mean = result != 0
        result_mean = tensorflow.boolean_mask(result,mask_mean)

        sortino_ratio = tensorflow.reduce_mean(result_mean) / tensorflow.math.reduce_std(result_mask) * math.sqrt(251)
        print("sortino ratio: ", sortino_ratio)
        return - sortino_ratio  # To minimize the loss function
    result = None
    for each in range(100):
        if result is None:
            result = model.predict(test_data)
        else:
            result = tensorflow.concat([result,model.predict(test_data)],1)

    """
    To make decision trade or not
    """
    mean = tensorflow.cast(tensorflow.math.reduce_mean(result,1),dtype=tensorflow.float32)
    mean = tensorflow.expand_dims(mean,axis = 1)
    std = tensorflow.cast(tensorflow.math.reduce_std(result,1),dtype=tensorflow.float32)
    std = tensorflow.expand_dims(std,axis = 1)
    abs_mean = tensorflow.math.abs(mean)
    decision = abs_mean > hyperpara_bayesian_decision * std
    zeros = tensorflow.zeros(decision.shape,dtype=tensorflow.float32)
    final_decision = tensorflow.raw_ops.Select(condition=decision,x=mean,y=zeros)
    #print(final_decision.shape)
    #print(final_decision)
    #final_decision = tensorflow.boolean_mask(final_decision, final_decision != 0)
    Backtest_Bay_Simple(y_true,final_decision)

    y_true_copy = y_true
    y_true = y_true[:,:16]
    y_true = y_true.astype(np.float64)
    call_position = tensorflow.ones(y_true.shape[0])
    put_position = tensorflow.abs(y_true[:, 4]) / tensorflow.abs(y_true[:, 12])  # Position amount = Call Delta / Put Delta as hedging
    call_earn = tensorflow.cast(call_position * (y_true[:, 3] - y_true[:, 2]), dtype=tensorflow.float64)
    put_earn = put_position * (y_true[:, 11] - y_true[:, 10])
    amount_money = tensorflow.cast(call_position * y_true[:, 2], dtype=tensorflow.float64) + put_position * y_true[:,10]
    result =  y_pred[:, 0] * ((call_earn + put_earn) / amount_money) - tensorflow.cast(tensorflow.abs(y_pred[:, 0]),dtype=tensorflow.float64) * float(0.005)  # Assume 0.5% trading fee

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
        earnnew = 0.1 * earn * (1 + each)
        earn = earn * 0.9 + earnnew
        earn_history.append(earn)
        #print(y_pred[n,0], -result[n], y_true[n, 0:2], y_true[n, 8:10],y_true_copy[n,16:18])
        if y_pred[n,0] > 0:
            buy += 1
        elif y_pred[n,0] < 0:
            sell += 1
        elif y_pred[n,0] == 0:
            no_action +=1
        n += 1
        #print(earn)
        if each < 0:
            lose += 1
        elif each > 0:
            win += 1

    print('win: ', win, "lose: ", lose)
    print('buy: ', buy, "sell: ", sell, 'no action: ', no_action)
    print('total earn: ', earn)
    print('withdraw: ', max_withdraw, max_withdraw_peak, max_withdraw_bottom)
    Sharpe_Ratio(y_true,y_pred)
    Sortino_Ratio_as_loss_fn(y_true,y_pred)
    label_name = str(loss[loss_num])+" Bayesian No Selection"
    plot_data[label_name] = earn_history
    return - earn
"""
Data Preprocessing
"""

# No Validation set for this one
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
    dataset = json.load(open("Training_Data/train_data.json",'r'))
    random.shuffle(dataset)
    dataset_np = np.array(dataset, dtype=object)
    dataset_np_train = np.array(dataset_np[:, 0].tolist())
    dataset_np_label = np.array(dataset_np[:, 1].tolist())

    output_train_data = dataset_np_train[:int(len(dataset_np_train) * 0.8)]
    output_train_label = dataset_np_label[:int(len(dataset_np_label) * 0.8)]
    output_test_data = dataset_np_train[int(len(dataset_np_label) * 0.8):]
    output_test_label = dataset_np_label[int(len(dataset_np_label) * 0.8):]

    return output_train_data, output_train_label, output_test_data, output_test_label


def Make_plot():
    """
    To make plot for the output
    :return: plots
    """
    for each in plot_data.keys():
        plt.plot(list(range(len(plot_data[each]))),plot_data[each],label=each)

    plt.title('The Trading result graph')
    plt.xlabel('times')
    plt.ylabel('total value')
    plt.legend()
    plt.show()

"""
Models
"""
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
    global loss, loss_num
    train_data = tensorflow.constant(train_data,dtype=tensorflow.float32)
    train_label = tensorflow.constant(train_label[:,:16],dtype=tensorflow.float32)
    test_data = tensorflow.constant(test_data,dtype=tensorflow.float32)
    inputs = Input(shape=(4,))
    benchmark_fc = Dense(3, activation='relu')(inputs)
    final_output = Dense(1, activation='tanh')(benchmark_fc)

    model = Model(inputs, final_output)
    """
    Hyperparameters: Loss function, Sample_weight_mode, batch_size, epochs...
    """
    model.compile(optimizer='adam', loss=loss[loss_num], run_eagerly=False)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(train_data, train_label,
              epochs=200,
              batch_size=20,
              # sample_weight=None,#initial_weight,
              shuffle=True),
    # callbacks=[early_stopping]),
    # validation_split=0.2)
    """
    model.save(
        filepath=model_save_path,
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        save_traces=True)
    """

    Backtest(test_label, model.predict(test_data))
    return

def Bayesian_Model(model_save_path,train_data, train_label, test_data, test_label):

    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tensorflow.zeros(n), scale_diag=tensorflow.ones(n)
                    )
                )
            ]
        )
        return prior_model

    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
                ),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

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
    global loss, loss_num
    class ScaleLayer(tensorflow.keras.layers.Layer):
        def __init__(self):
          super(ScaleLayer, self).__init__()
          self.scale = tensorflow.Variable(1.,trainable=True,dtype=tensorflow.float32)

        def call(self, inputs1, inputs2):
            if inputs1 > 0:
                return inputs1 - inputs2 * self.scale
            else:
                return inputs1 + inputs2 * self.scale

    class DecisionLayer(tensorflow.keras.layers.Layer):
        def __init__(self):
          super(DecisionLayer, self).__init__()

        def call(self, inputs1, inputs2):
            a = inputs1[0] > 0
            c = inputs2[0] > 0
            if a == c:
                return inputs2
            else:
                if a:
                    return tensorflow.keras.layers.ReLU()(inputs2 - inputs1)
                else:
                    return tensorflow.keras.layers.ReLU()(inputs1 - inputs2)
    class Monte_Carlo_layer(tensorflow.keras.layers.Layer):
        def __init__(self,layer):
          super(Monte_Carlo_layer, self).__init__()
          self.layer = layer

        def call(self,inputs):
            sample = None
            for each in range(100):
                final_output = self.layer(inputs)
                if sample == None:
                    sample = tensorflow.cast(final_output,dtype=tensorflow.float32)
                else:
                    final_output = tensorflow.cast(final_output,dtype=tensorflow.float32)
                    sample = tensorflow.concat([sample,final_output],1)
            std = tensorflow.math.reduce_std(sample,1)
            mean = tensorflow.reduce_mean(sample,1)

            std = tensorflow.expand_dims(std,axis = 1)
            mean = tensorflow.expand_dims(mean,axis = 1)
            concat = tensorflow.concat([mean, std],1)
            return concat
    train_data = tensorflow.constant(train_data,dtype=tensorflow.float32)
    train_label = tensorflow.constant(train_label[:,:16],dtype=tensorflow.float32)
    test_data = tensorflow.constant(test_data,dtype=tensorflow.float32)
    inputs = Input(shape=(4,))
    features = tfp.layers.DenseVariational(
        units=3,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / (train_data.shape[0] / 20),
        activation="sigmoid",
    )(inputs)
    # benchmark_fc = Dense(3, activation='relu')(features)

    #final_final_output = Monte_Carlo_layer(layer)(features)
    final_final_output = Dense(1, activation='tanh')(features)
    #final_final_output = Dense(1,activation='tanh')(output)


    model = Model(inputs,final_final_output)
    """
    Hyperparameters: Loss function, Sample_weight_mode, batch_size, epochs...
    """
    model.compile(optimizer='adam', loss=loss[loss_num], run_eagerly=False)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(train_data, train_label,
              epochs=400,
              batch_size=20,
              # sample_weight=None,#initial_weight,
              shuffle=True),
    # callbacks=[early_stopping]),
    # validation_split=0.2)
    #model.save_weights(model_save_path)

    Backtest_Bayesian(test_label, model.predict(test_data), model, test_data,0)
    return model



hyperpara_bayesian_decision = 1
loss = [Sharpe_Ratio_as_loss_fn,Sortino_Ratio_as_loss_fn,Volatility_only_expected_profit_loss_fn]
loss_num = 1
plot_data = {}

train_data, train_label, test_data, test_label = Data_Preprocessing()


num = 1
TOTAL_EARN = []
WITHDRAW = []
SHARPE = []
SORTINO = []

# Result 0: Get data shuffle back test result //
# Test for the random seed and shuffle to check whether the performance is by accidnet or not.
def data_shuffle():
    for each in range(50):
        random.seed(each)
        train_data, train_label, test_data, test_label = Data_Preprocessing()
        Easy_Model('No save',train_data, train_label, test_data, test_label)

# Result 1: For model and loss function compare
def result_1():
    global loss_num
    for each in range(3):
        loss_num = each
        Easy_Model('No save',train_data, train_label, test_data, test_label)
        Bayesian_Model('No save',train_data, train_label, test_data, test_label)



# Result 2: For Bayesian model with different confirmation condition
def result_2():
    global hyperpara_bayesian_decision
    model = Bayesian_Model('No save',train_data, train_label, test_data, test_label)
    y_pred = model.predict(test_data)
    for each in range(2,11):
        hyperpara_bayesian_decision = 0.5 * each
        Backtest_Bayesian(test_label, y_pred, model, test_data, 0)

print('which result to test? 0 for data shuffle test; 1 for result 1; 2 for result 2:')
a = input("input:")
if int(a) == 0:
    data_shuffle()
    Make_plot()
elif int(a) == 1:
    result_1()
    Make_plot()
elif int(a) == 2:
    result_2()
    Make_plot()
else:
    print('Invalid input!!')


