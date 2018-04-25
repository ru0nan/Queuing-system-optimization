# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:31:40 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 08:51:09 2018

@author: lenovo
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:33:27 2018
结合排队模型寻优参数
使得超出时间最小
@author: lenovo 
"""

# !/usr/bin/env python  
# -*- coding:utf-8 -*-   
# Author: wsw  
# 简单实现SGA算法  
import numpy as np 
import math 
from scipy.optimize import fsolve
import random  
import timeit 
import copy
import heapq
import matplotlib.pyplot as plt 
from imp import reload 
import queModel_inter_wtis as qm  
  
  
# 根据解的精度确定染色体(chromosome)的长度  
# 需要根据决策变量的上下边界来确定  
def getEncodedLength(delta=0.001, boundarylist=[]):  
    # 每个变量的编码长度  
    lengths = []  
    for i in boundarylist:  
        lower = i[0]  
        upper = i[1]  
        # lamnda 代表匿名函数f(x)=0,50代表搜索的初始解  
        res = fsolve(lambda x: ((upper - lower) * 1 / delta) - 2 ** x + 1, 50)  
        length = int(np.ceil(res[0]))  
        lengths.append(length)  
    return lengths  
    pass  
  
  
# 随机生成初始编码种群  
def getIntialPopulation(encodelength, populationSize):  
    # 随机化初始种群为0  
    chromosomes = np.zeros((populationSize, sum(encodelength)), dtype=np.uint8)  
    for i in range(populationSize):  
        chromosomes[i, :] = np.random.randint(0, 2, sum(encodelength))  
    # print('chromosomes shape:', chromosomes.shape)  
    return chromosomes  
  
  
# 染色体解码得到表现型的解  
def decodedChromosome(encodelength, chromosomes, boundarylist, delta=0.001):  
    populations = chromosomes.shape[0]  
    variables = len(encodelength)  
    decodedvalues = np.zeros((populations, variables))  
    for k, chromosome in enumerate(chromosomes):  
        chromosome = chromosome.tolist()  
        start = 0  
        for index, length in enumerate(encodelength):  
            # 将一个染色体进行拆分，得到染色体片段  
            power = length - 1  
            # 解码得到的10进制数字  
            demical = 0  
            for i in range(start, length + start):  
                demical += chromosome[i] * (2 ** power)  
                power -= 1  
            lower = boundarylist[index][0]  
            upper = boundarylist[index][1]  
            decodedvalue = lower + demical * (upper - lower) / (2 ** length - 1)  
            decodedvalues[k, index] = decodedvalue  
            # 开始去下一段染色体的编码  
            start = length  
    return decodedvalues  
  
  
# 得到个体的适应度值及每个个体被选择的累积概率  
def getFitnessValue(func, chromosomesdecoded):
    
    # 得到种群规模和决策变量的个数  
    population, nums = chromosomesdecoded.shape 

    #最差的个数及下标
    worst = 0
    worst_index = np.zeros((worst),dtype = np.uint8)
    # 初始化种群的适应度值为0  
    objvalues = np.zeros((population, 1))  
    fitnessvalues = np.zeros((population, 1))  
    # 计算适应度值
    low_bound = 3 #为了扩大范围而减去的天数
    for i in range(population):  
        objvalues[i, 0] = func(chromosomesdecoded[i, :]) 
        if(objvalues[i, 0]<low_bound):
            print("\n\n!!!!!!!!!小于%d天了!!!!!!!!!!!\n" %low_bound)
        #fitnessvalues[i,0] = 2**(100000/10**(objvalues[i, 0] -low_bound))
        fitnessvalues[i,0] = 1000/1.5**(objvalues[i,0])
        print('参数{}  适应度{}'.format(chromosomesdecoded[i, :],fitnessvalues[i,0]))
    #寻找精英
    elite_value = np.max(fitnessvalues[:,0])
    elite_index = np.where(fitnessvalues[:,0] == elite_value)
        
    #去掉最差的
    if worst>0:
        temp = heapq.nsmallest(worst,fitnessvalues)
        for i in range(worst):
            worst_index = np.where(fitnessvalues == temp[i])
            fitnessvalues[worst_index[0][0]] = 0        
    
    # 计算每个染色体被选择的概率  
    probability = fitnessvalues / np.sum(fitnessvalues)  
    # 得到每个染色体被选中的累积概率  
    cum_probability = np.cumsum(probability)  

    return fitnessvalues, cum_probability, elite_index 
  
  
# 新种群选择  
def selectNewPopulation(chromosomes, cum_probability, elite_index):  
    select_index=[]
    m, n = chromosomes.shape  
    newpopulation = np.zeros((m, n), dtype=np.uint8)
    #前elite个染色体是精英
    for i in range(1):
        newpopulation[i] = chromosomes[elite_index[i][0]]
        select_index.append(elite_index[i][0])

    count = 1 #新种群中旧精英的个数
    limit = 4 #新种群中旧精英的最大个数
    i = 0
    while i < m-1:
        randoma = np.random.rand()
        logical = cum_probability >= randoma  
        index = np.where(logical == 1)
        # index是tuple,tuple中元素是ndarray
        if count<limit:
            newpopulation[i+1, :] = chromosomes[index[0][0], :]
            select_index.append(index[0][0])
            i += 1
            if index[0][0] in elite_index[0]:
                count += 1
        elif not(index[0][0] in elite_index[0]):
            newpopulation[i+1, :] = chromosomes[index[0][0], :]
            select_index.append(index[0][0])
            i += 1
        else:
            pass
    #print('count = ',count)
            
    return newpopulation, select_index
  
  
# 新种群交叉  
def crossover(population,  fitnessvalues):
    k1 = 1
    k3 = 1
    updatepopulation = copy.deepcopy(population)
    n_indv,len_indv = population.shape
    f_max = np.max(list(fitnessvalues))
    f_avg = np.mean(list(fitnessvalues))
    cross_index = [x for x in range(n_indv)]
    random.shuffle(cross_index)
    for i in range(0,n_indv-1,2):
        f_apo = max([fitnessvalues[cross_index[i]][0],fitnessvalues[cross_index[i+1]][0]])
        if f_apo < f_avg:
            Pc = k3
        else:
            Pc = k1*(f_max-f_apo)/(f_max-f_avg)
        flag = random.random()
        #print('Pc=%4f,   flag=%4f'%(Pc,flag))
        if flag < Pc:
            a = cross_index[i]
            b = cross_index[i+1]
            #print('%d  %d  cross'%(a,b))
            # 随机产生2个交叉点  
            crossoverPoint = random.sample(range(1, len_indv), 2)
            #升序排列交叉结点
            crossoverPoint = sorted(crossoverPoint)  
            #两点交叉
            updatepopulation[a,crossoverPoint[0]:crossoverPoint[1]] = population[b,crossoverPoint[0]:crossoverPoint[1]]
            updatepopulation[b,crossoverPoint[0]:crossoverPoint[1]] = population[a,crossoverPoint[0]:crossoverPoint[1]]  
    
    return updatepopulation

# 染色体变异  
def mutation(population,  fitnessvalues):  
    """ 
 
    :param population: 经交叉后得到的种群 
    :param Pm: 变异概率默认是0.01 
    :return: 经变异操作后的新种群 
    """ 
    k2 = 0.5
    k4 = 0.5
    updatepopulation = copy.deepcopy(population)
    n_indv,len_indv = population.shape
    f_max = np.max(list(fitnessvalues))
    f_avg = np.mean(list(fitnessvalues))
    f_min = np.min(list(fitnessvalues))

    for i in range(n_indv):
        if fitnessvalues[i][0]<f_avg:
            Pm = k2
        else :
            Pm = k4*(f_max-fitnessvalues[i][0])/(f_max-f_avg)
        # 计算需要变异的基因个数  
        gene_num = int(round(len_indv*(f_max-fitnessvalues[i][0])/(f_max-f_min)/3))
        flag = random.random()
        #print('Pm=%4f,   flag=%4f'%(Pm,flag))
        if flag < Pm:
            #print('%d  mutation'%i)
            # 随机抽取gene_num个基因进行基本位变异  
            mutationGeneIndex = random.sample(range(0, len_indv), gene_num)  
            # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)  
            for gene in mutationGeneIndex:  
                # 确定变异基因位于当前染色体的第几个基因位  
                # mutation  
                if updatepopulation[i, gene] == 0:  
                    updatepopulation[i, gene] = 1  
                else:  
                    updatepopulation[i, gene] = 0  
                    
    return updatepopulation  
  
# 定义目标函数  
def objFunc():  
    return lambda x: qm.output(x[0],x[1])

#    return lambda x: 20 + np.square(x[0]) +np.square(x[1]) \
#    - 10*np.cos(2*np.pi*x[0]) - 10*np.cos(2*np.pi*x[1])
#    return lambda x: 21.5 + x[0] * np.sin(4 * np.pi * x[0]) + x[1] * np.sin(20 * np.pi * x[1])  
  
  
#def main(max_iter=300):  
if __name__ ==  '__main__':
    max_iter = 80
    popSize = 10
    # 每次迭代得到的最优解  
    optimalSolutions = []  
    optimalValues = []  #最优目标函数值
    optimalFitness = [] #最优适应度值
    avgFitness = [] #平均适应度值
    # 决策变量的取值范围  
    decisionVariables = [[0, 10], [0, 1]]  
    # 得到染色体编码长度  
    lengthEncode = getEncodedLength(boundarylist=decisionVariables)  
    # 得到初始种群编码  
    chromosomesEncoded = getIntialPopulation(lengthEncode, popSize)
    fitnessvalues = np.zeros((popSize,1))
    #print('初始化\n',chromosomesEncoded)
    
    #print('累计概率\n',cum_proba)
    for iteration in range(max_iter): 
        # 种群解码  
        decoded = decodedChromosome(lengthEncode, chromosomesEncoded, decisionVariables)
        # 得到个体适应度值和个体的累积概率  
        fitnessvalues_before, cum_proba,elite_index  = getFitnessValue(objFunc(), decoded)  
        print('elite ',elite_index)

        # 选择新的种群  
        newpopulations,select_index = selectNewPopulation(chromosomesEncoded, cum_proba,elite_index)  
        for i,value in enumerate(select_index):
            fitnessvalues[i] = fitnessvalues_before[value]

        #交叉
        crossoverpopulation = crossover(newpopulations, fitnessvalues)  
#        print('交叉')
        # 变异
        mutationpopulation = mutation(crossoverpopulation, fitnessvalues)  
#        print('变异')

        # 将变异后的种群解码，得到每轮迭代最终的种群  
        final_decoded = decodedChromosome(lengthEncode, mutationpopulation, decisionVariables)  


        # 适应度评价  
        #fitnessvalues, cum_individual_proba,elite_index = getFitnessValue(objFunc(), final_decoded)  
        #print('适应度')
        
         
        # 搜索每次迭代的最大适应度值
        optimalFitness.append(np.max(list(fitnessvalues)))
        #反求最优（最小）目标函数值
        #optimalValues.append(math.log10((100000/math.log2(optimalFitness[-1])))+3) 
        optimalValues.append(math.log(1000/optimalFitness[-1],1.5) )
        print('%d iter,   best value: %4f'%(iteration,optimalValues[-1]))

        #平均适应度值
        avgFitness.append(np.mean(list(fitnessvalues)))        
        #最优解
        index = np.where(fitnessvalues == max(list(fitnessvalues)))  
        optimalSolutions.append(final_decoded[index[0][0], :])  

        #update
        chromosomesEncoded = mutationpopulation
        #cum_proba = cum_individual_proba
        
    # 搜索最优解  
    optimalValue = np.min(optimalValues)  
    optimalIndex = np.where(optimalValues == optimalValue)  
    optimalSolution = optimalSolutions[optimalIndex[0][0]] 
    
    plt.figure(1)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.plot(optimalValues)

  
  
## 测量运行时间  
#elapsedtime = timeit.timeit(stmt=main, number=1)  
#print('Searching Time Elapsed:(S)', elapsedtime)  

