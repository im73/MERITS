import json
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

with open("./data/data_glu_action_lab_basic_diagnose_final.json", "r", ) as f:
    dataset = json.load(f)

day_length = 25


def timecal(time1, time2):
    time1 = datetime.strptime(time1, '%m/%d/%Y')
    time2 = datetime.strptime(time2, '%m/%d/%Y')

    timedelta = str(time2 - time1)
    diff = abs(int(timedelta.split(' ')[0]))
    if (diff == 1) | (diff == 2):
        return 1
    else:
        return 0

def DataLoader(min_len = 3):
    whole_set =  []
    for key in dataset:
        data = []

        date_last = ''

        for date in dataset[key]:
            glu, time, action, lab = dataset[key][date]
            '''
            glu padding
            '''
            glu = glu[:min(len(glu), 25)]
            time = time[:min(len(time), 25)]
            temp_length = len(glu)
            for i in range(min(len(glu), 25)):
                '''
                nomalize for time : minutes / (24 * 60)
                '''

                time[i] = time[i] / 1440


            for i in range(25 - temp_length):
                time.append(0)
                glu.append(0)

            if date_last == '':

                date_last = date
                data.append([lab, glu, time, action, temp_length])

            elif timecal(date_last, date) and (len(data) < 25):
                date_last = date

                if sum(action)==0:
                    action = data[-1][3].copy()
                data.append([lab, glu, time, action, temp_length])

            else:
                '''
                add previous glu 
                '''

                for i in range(len(data)):

                    if i == 0:
                        data[i][1] = [data[i][1].copy()]
                        data[i][2] = [data[i][2].copy()]
                        data[i][3] = [data[i][3].copy()]
                        data[i][4] = [data[i][4]]

                    else:
                        data[i][1] = data[i-1][1] + [data[i][1]]
                        data[i][2] = data[i-1][2] + [data[i][2]]
                        data[i][3] = data[i-1][3] + [data[i][3]]
                        data[i][4] = data[i-1][4] + [data[i][4]]
                whole_set += data

                data = [[lab, glu, time, action, temp_length]]

                date_last = date

        '''
        add previous glu time
        '''

        for i in range(len(data)):

            if i == 0:
                data[i][1] = [data[i][1].copy()]
                data[i][2] = [data[i][2].copy()]
                data[i][3] = [data[i][3].copy()]
                data[i][4] = [data[i][4]]

            else:
                data[i][1] = data[i-1][1] + [data[i][1]]
                data[i][2] = data[i-1][2] + [data[i][2]]
                data[i][3] = data[i-1][3] + [data[i][3]]
                data[i][4] = data[i-1][4] + [data[i][4]]

        whole_set += data

    fileter_set = []
    index = 0
    for i in range(len(whole_set)):
        if len(whole_set[i][4]) != len(whole_set[i][3]):
            
            index +=1
        if len(whole_set[i][4]) == 1:
            continue
        elif len(whole_set[i][4]) > min_len:
            fileter_set.append(whole_set[i])
            
    return fileter_set


def Mul_step_DataLoader(min_len):
    whole_set = []
    num_p = 0
    for key in dataset:
        data = []

        date_last = ''
        num_p += 1
        for date in dataset[key]:
            glu, time, action, lab = dataset[key][date]
            '''
            glu padding
            '''
            glu = glu[:min(len(glu), 25)]
            time = time[:min(len(time), 25)]
            temp_length = len(glu)
            for i in range(min(len(glu), 25)):
                '''
                nomalize for time : minutes / (24 * 60)
                '''

                time[i] = time[i] / 1440

            for i in range(25 - temp_length):
                time.append(0)
                glu.append(0)

            if date_last == '':

                date_last = date
                data.append([lab, glu, time, action, temp_length])

            elif timecal(date_last, date) and (len(data) < 25):
                date_last = date

                if sum(action) == 0:
                    action = data[-1][3].copy()
                data.append([lab, glu, time, action, temp_length])

            else:
                '''
                add previous glu 
                '''

                for i in range(len(data)):

                    if i == 0:
                        data[i][1] = [data[i][1].copy()]
                        data[i][2] = [data[i][2].copy()]
                        data[i][3] = [data[i][3].copy()]
                        data[i][4] = [data[i][4]]

                    else:
                        data[i][1] = data[i - 1][1] + [data[i][1]]
                        data[i][2] = data[i - 1][2] + [data[i][2]]
                        data[i][3] = data[i - 1][3] + [data[i][3]]
                        data[i][4] = data[i - 1][4] + [data[i][4]]
                whole_set += data
                


                data = [[lab, glu, time, action, temp_length]]

                date_last = date

        '''
        add previous glu time
        '''

        for i in range(len(data)):

            if i == 0:
                data[i][1] = [data[i][1].copy()]
                data[i][2] = [data[i][2].copy()]
                data[i][3] = [data[i][3].copy()]
                data[i][4] = [data[i][4]]

            else:
                data[i][1] = data[i - 1][1] + [data[i][1]]
                data[i][2] = data[i - 1][2] + [data[i][2]]
                data[i][3] = data[i - 1][3] + [data[i][3]]
                data[i][4] = data[i - 1][4] + [data[i][4]]

        whole_set += data
        


    fileter_set = []
    index = 0
    for i in range(len(whole_set)):
        if len(whole_set[i][4]) != len(whole_set[i][3]):
            index += 1
        if len(whole_set[i][4]) == 1:
            continue
        elif len(whole_set[i][4]) >= min_len:
            fileter_set.append(whole_set[i])

    return fileter_set

