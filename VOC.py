#import jieba
import pickle
import argparse
import os
import re
from gensim.models import word2vec
import datetime
result = []
result1 = []
epoch = []
mmAp = []
maxmAp = []
ccAp = []
maxccAp = []
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', dest='path', type=str, default='vg_graph_r_a.pkl',
                    help='an integer for the accumulator')


args = parser.parse_args()
path=os.getcwd() + '/'+ args.path

with open(path, "r", encoding="utf-8") as f_reader:
    for line in f_reader:
        check_epoch = re.findall(r'/cascade_fpn_1_(.*)_', line)
        #check_epoch = re.findall(r'/pascal_voc_07_baseline_1_(.*)_2327.pth', line)
        #check_epoch = re.findall(r'cfg_file=\'cfgs/res101_ms.yml\', checkepoch=(.*), checkpoint', line)
        if check_epoch != []:
            #print(check_epoch)
            epoch.append(check_epoch[0])
        #AP = re.findall(r'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = (.*)', line)
        clamp = re.findall(r'clamp = (.*)', line)
        if clamp != []:
            result.append(clamp[0])
            #print("clamp = "+ str(clamp))
        board = re.findall(r'board = (.*)', line)
        if board != []:
            result.append(board[0])
            #print("board = "+ str(board))
        ring = re.findall(r'ring = (.*)', line)
        if ring != []:
            result.append(ring[0])
            #print("ring = " + str(ring))
        plate = re.findall(r'plate = (.*)', line)
        if plate != []:
            result.append(plate[0])
            #print("plate = " + str(plate))
        clip = re.findall(r'clip = (.*)', line)
        if clip != []:
            result.append(clip[0])
            #print("clip = " + str(clip))
        hammer = re.findall(r'hammer = (.*)', line)
        if hammer != []:
            result.append(hammer[0])
            #print("hammer = " + str(hammer))
        spacer  = re.findall(r'spacer = (.*)', line)
        if spacer  != []:
            result.append(spacer[0])
            #print("spacer = " + str(spacer))
        weight = re.findall(r'weight = (.*)', line)
        if weight != []:
            result.append(weight[0])
            #print("weight = " + str(weight))
        AP = re.findall(r'Mean AP = (.*)', line)
        if AP != []:
            result.append(AP[0])
            # print("weight = " + str(weight))

        AP = re.findall(r'maxDets=100 ] = (.*)', line)
        if AP != []:
            result1.append(AP[0])
        AP1 = re.findall(r'maxDets=  1 ] = (.*)', line)
        if AP1 != []:
            result1.append(AP1[0])
            # print("AP1 = "+ str(AP1))
        AP10 = re.findall(r'maxDets= 10 ] = (.*)', line)
        if AP10 != []:
            result1.append(AP10[0])
            # print("AP10 = " + str(AP10))
            
    for i in range(len(epoch)):
        tmp_result1 = []
        tmp_result1.append(epoch[i])
        for j in result1[i*12:12*i+12]:
            tmp_result1.append(str(round(float(j)*100,2)))
        ccAp.append(tmp_result1)
        maxccAp.append(ccAp[i][2])
 ##------------------------------------------------------------------------
    ccAp1 = ccAp[maxccAp.index(max(maxccAp))]
    ccAp2 = ccAp1.copy()
    result2 = ''
    for i in range(len(ccAp2)):
        result2 = result2 + ' ' + ccAp2[i]
        

    for i in range(len(epoch)):
        tmp_result = []
        tmp_result.append(epoch[i])
        for j in result[i*15:15*i+15]:
            tmp_result.append(str(round(float(j)*100,2)))
        mmAp.append(tmp_result)
        maxmAp.append(mmAp[i][-1])
        #print("epoch" + epoch[i]+ tmp_result)

    mmAp1 = mmAp[maxmAp.index(max(maxmAp))]
    mmAp2 = mmAp1.copy()
    mmAp2[1] = mmAp1[-1]
    for i in range(1,len(mmAp1)-1):
        mmAp2[i+1] = mmAp1[i]
    result0 = ''
    for i in range(len(mmAp2)):
        result0 = result0 + ' ' + mmAp2[i]

    print(result0)
    print(result2)

    parameter = args.path.split('_')
    # print(parameter)
    #learning = parameter[-3]
    learning = ''
    #decay = parameter[-2]
    decay = ''
    out_result = 'Resnet101' + ' ' + 'cascade_hkrm_333001' + ' ' + 'CharmingWang' + ' ' + \
                 str(datetime.date.today()) + ' ' + '1 ' + learning + ' ' + decay + ' ' + args.path  +result0
    out_result1 = ' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' + str(ccAp2[1]) + "\r\n" + \
        ' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ' + str(ccAp2[2]) + "\r\n" + \
        ' Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ' + str(ccAp2[3]) + "\r\n" + \
        ' Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ' + str(ccAp2[4]) + "\r\n" + \
        ' Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ' + str(ccAp2[5]) + "\r\n" + \
        ' Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ' + str(ccAp2[6]) + "\r\n" + \
        ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ' + str(ccAp2[7]) + "\r\n" + \
        ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ' + str(ccAp2[8]) + "\r\n" + \
        ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' + str(ccAp2[9]) + "\r\n" + \
        ' Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ' + str(ccAp2[10]) + "\r\n" + \
        ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ' + str(ccAp2[11]) + "\r\n" + \
        ' Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ' + str(ccAp2[12]) + "\r\n"


    print(out_result)
    print(out_result1)




f= open("testresult.txt","a+")
f.write(out_result + "\r\n")
f.write(out_result1 + "\r\n")
f.close()
