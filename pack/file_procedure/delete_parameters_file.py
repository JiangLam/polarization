# Author: zhou xu
# Function: 删除已经计算好的偏振参数文件【包括.mat\.jpg\以及原始偏振数据等】，只保留file_path变量中的文件，并将新的复制文件名后面加入‘_’以示区别
# Goal: 删除不使用的文件，减少空间占用
# Information:  dir_path [字符串，要删除文件夹的上一级文件目录，如想删除的文件夹名为‘/1‘，则dir_path应为’/b/1‘中的到’/b‘的目录]
#               file_path [列表，保留想要的文件，可自行添加、删除]


import os
import shutil
import time


def deleter_parameters(dir_path):
    '''删除已经计算好的偏振参数文件，
       只保留file_path中的文件，
       将新的文件名后加‘_’以示区别'''
    file_path = ['/FinalMM.jpg','/FinalMM.mat','/FinalMM_[-0.1 0.1].jpg','/m11.jpg','/m11.mat']
    print(dir_path, end='     ')
    begin = time.time()
    for s in os.listdir(dir_path + '.'):
        path = dir_path + '/' + s + '_'
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            continue
        for o in file_path:
            path_ori = dir_path + '/' + s + o
            shutil.copy(path_ori, path)
        dir_path_del = dir_path + '/' + s
        shutil.rmtree(dir_path_del)
    end = time.time()
    spend = end - begin
    print(spend)