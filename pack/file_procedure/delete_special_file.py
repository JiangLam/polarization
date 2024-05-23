# Author: zhou xu
# Function: 删除特定文件
# Goal: 删除不必要文件，减少空间占用
# Information:  dir_path [要删除文件的文件夹的上一级，如要删除’/1/m11.mat‘，则dir_path应为’a/1/..'中的‘/a’str路径]
#               file_name [str，需要输入完整、正确的文件名，否则无法删除]


import os


def delete_special_file(dir_path, file_name):
    ''' 删除特定的一个不必要文件，以减少空间占用'''
    for i in os.listdir(dir_path):
        path = dir_path + '/' + i
        file_list = os.listdir(path)
        if file_name in file_list:
            file_path = path + '/' + file_name
            print(file_path + '  has been deleted')
            os.remove(file_path)

if __name__ == '__main__':
    dir_path = 'G:/第3组毛囊样本数据/WSI/D1R'
    file_name = 'sdf.txt'
    delete_special_file(dir_path, file_name)