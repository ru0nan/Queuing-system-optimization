# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:35:07 2018

@author: lenovo
"""

import numpy as np 
import time 
import matplotlib.pyplot as plt  
import queModel_interface as que
from mpl_toolkits.mplot3d import Axes3D  
  
class PSO(object):  
    def __init__(self, population_size, max_steps):  
        self.w = 0.6  # 惯性权重  
        self.c1 = self.c2 = 2  
        self.population_size = population_size  # 粒子群数量  
        self.dim = 2  # 搜索空间的维度  
        self.max_steps = max_steps  # 迭代次数  
        self.x_bound = [0, 16, 0, 2]  # 解空间范围 
        x1 = np.random.uniform(self.x_bound[0], self.x_bound[1],  
                                   (self.population_size, 1))  # 初始化粒子群位置
        x2 = np.random.uniform(self.x_bound[2], self.x_bound[3],  
                                   (self.population_size, 1))  # 初始化粒子群位置
        self.x = np.append(x1,x2,axis=1)
        
#        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],  
#                                   (self.population_size, self.dim))  # 初始化粒子群位置  
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度  
        fitness = self.calculate_fitness(self.x)  
        self.p = self.x  # 个体的最佳位置  
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置  
        self.individual_best_fitness = fitness  # 个体的最优适应度  
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度  
  
    def calculate_fitness(self, x):
        len_x = len(x)
        fitness = np.zeros((len_x,1))
        for i in range(len_x):
            fitness[i] = que.output(x[i,0],x[i,1])
        return fitness
    
#        with local optimal
#        fitness= 20 + np.sum(np.square(x), axis=1) \
#        - 10*np.cos(0.3*np.pi*x[:,0]) - 10*np.cos(0.3*np.pi*x[:,1])
#        return fitness
        
#        without local optimal
#        return np.sum(np.square(x), axis=1)  

    def lim(self):
    #位置限制
        for i in range(self.dim):
            aux1 = np.arange(self.population_size)#辅助数组
            #下限限制
            aux2 = aux1[self.x[:,i]<self.x_bound[i*self.dim]]#辅助数组，返回所有值满足条件的数组下标
            self.x[aux2,i] = self.x_bound[i*self.dim]
            #上限限制
            aux2 = aux1[self.x[:,i]>self.x_bound[i*self.dim+1]]
            self.x[aux2,i] = self.x_bound[i*self.dim+1]
        
    def evolve(self):  
        
# =============================plot initial====================================
#         fig = plt.figure()
#         plt.axis([-10, 10,-10,10])
#         ax = Axes3D(fig)
#         X = np.arange(-10, 10, 0.25)
#         Y = np.arange(-10, 10, 0.25)
#         X_ , Y_ = np.meshgrid(X, Y)
#         Z_= 20 + np.square(X_) + np.square(Y_) \
#         - 10*np.cos(0.3*np.pi*X_) - 10*np.cos(0.3*np.pi*Y_)
# #        Z_ = np.square(X_)+np.square(Y_)
# =============================================================================
        for step in range(self.max_steps):  
            r1 = np.random.rand(self.population_size, self.dim)  
            r2 = np.random.rand(self.population_size, self.dim)  
            # 更新速度和权重  
            self.v = self.w*self.v+self.c1*r1*(self.p-self.x)+self.c2*r2*(self.pg-self.x)  
            self.x = self.v + self.x
            #限制位置
            self.lim()
                
            fitness = self.calculate_fitness(self.x)  

# ================================plot update==================================
#             ax.cla()
#             ax.set_xlim(-10,10)
#             ax.set_ylim(-10,10)
#             ax.set_zlim(0,200)
#             ax.plot_wireframe(X_, Y_, Z_, rstride=5, cstride=5) 
# #            ax.plot_surface(X_, Y_, Z_, rstride=1, cstride=1, cmap='rainbow')
#             ax.scatter(self.x[:,0],self.x[:,1],fitness,c='k')
#             
# =============================================================================

#            plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')  
#            plt.xlim(self.x_bound[0], self.x_bound[1])  
#            plt.ylim(self.x_bound[0], self.x_bound[1])  
#            plt.pause(0.01)  

            # 需要更新的个体  
            aux1 = np.arange(self.population_size)
            aux2 = np.greater(self.individual_best_fitness, fitness)  
            aux2 = np.transpose(aux2)
            update_id = aux1[aux2[0]]
#            print('index shape',update_id.shape)
#            print('self.p shape',self.p.shape)
#            
            self.p[update_id,0] = self.x[update_id,0]
            self.p[update_id,1] = self.x[update_id,1]
            self.individual_best_fitness[update_id] = fitness[update_id]  
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置  
            if np.min(fitness) < self.global_best_fitness:  
                self.pg = self.x[np.argmin(fitness)]  
                self.global_best_fitness = np.min(fitness)
            trace[step,] = [self.global_best_fitness, np.mean(fitness)]
            
            print('best fitness: %d, mean fitness: %.3f' % (trace[step,0], trace[step,1]))  
  

pso = PSO(40, 60)
trace = np.zeros((pso.max_steps,2))   
pso.evolve()  
plt.show() 