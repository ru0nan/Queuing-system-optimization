# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:15:55 2018
用openpyxl 将txt转为excel ，提取MRI_Booking2014中4107医院的患者到xlsx
再将他写入excel中去
@author: lenovo
"""

from openpyxl import Workbook
readFile = 'D:\\学习\\毕设\\数据资料\\MRI Data\\MRI_Patient.txt'
writeFile = 'D:\\学习\\毕设\\数据资料\\MRI Data\\4107_Patient.xlsx'
flag = False
i = 1
wb = Workbook()
sheet = wb.active
#print(wb.get_sheet_names()) # 提供一个默认名叫Sheet的表，office2016下新建提供默认Sheet1
# 直接赋值就可以改工作表的名称
sheet.title = 'Sheet1'
# 新建一个工作表，可以指定索引，适当安排其在工作簿中的位置
#wb.create_sheet('Sheet1', index=1) # 被安排到第二个工作表，index=0就是第一个位置
#f = open(readFile,'r')   #打开txt文本进行读取
with open(readFile,'r') as f:
    for line in f:
        string1 = line.split('|')
        if flag == False:
           sheet.append(string1)
        if flag and (int(string1[0]) == 4107):
            sheet.append(string1)           
            i +=1
        flag = True
        
#while True:  #循环，读取文本里面的所有内容
#    line = f.readline() #一行一行读取
#    if not line:  #如果没有内容，则退出循环
#        break
#
#    line = line.split('\t')
#    sheet.append(line)  


wb.save(writeFile)