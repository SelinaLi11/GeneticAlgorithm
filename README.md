# GeneticAlgorithm
A demo of GA framework to solve Knapsack problem


进化算法是一类算法的总称。这类算法的灵感多来源于自然界中的进化现象。遗传算法是最基本的进化算法之一。本文首先引入背包问题，使用遗传算法解决该问题，以此介绍遗传算法的基本概念。

### 01背包问题

[背包问题九讲](https://github.com/tianyicui/pack)

#### 问题描述

现有一个背包，其容量上限为1000，有50个物品可以放进该背包中，每个物品占容量不同，有不同的价值，求在背包容量范围内如何放置物品使得总价值最高。

#### 思路

在遗传算法中，问题的可行解被视为**个体**，多个可行解的集合被视为**种群**。种群会经历一代代的演化，在每一代中，需要通过**选择算子**选出（一定父代（现有解）进行繁殖，产生子代（新解），在繁殖的过程中，父代染色体会发生**重组**（将两个解的放置方式进行重组）和**突变**（改变解中某个物品的放置方式）以产生子代。

我们如何求解背包问题呢？首先，我们需要定义**解**（个体/染色体）的形式，在背包问题中，解的意义即为物品的放置方式，可以表示为010100111....，其中，0代表对应的**物品**（基因）不放入背包，1代表对应的**物品**（基因）放入背包。我们如何找到总价值最高的解呢？对每个解（个体/染色体）计算其**总价值**（适应度），总价值（适应度）最高的解（个体/染色体）即为目标解。

p.s. 遗传算法不是精确算法，最终解不一定为最优解。

* 遗传算法基本流程

![遗传算法基本流程](https://tva1.sinaimg.cn/large/007S8ZIlly1ghemckuex2j30jo0f8jry.jpg)

背包问题天然适合使用遗传算法来解。因为它的解可以使用二进制列进行表示，如果解是**实数或其他形式**（表现型），一般需要进行编码转化为**二进制列**（基因型），相应在算法中就需要定义编码函数和解码函数，这部分在本文中暂不涉及。

### 代码

本部分先展示核心代码，而后给出各类的详细描述。

#### 核心代码

```python
import numpy as np
import pandas as pd
import random
import itertools
import time

# 常数定义
weight_MAX = 1000  # 背包的最大容量
weights = [80,82,85,70,72,70,66,50,55,25,50,55,40,48,50,32,22,60,30,32,40,38,35,32,25,28,30,22,25,30,45,30,60,50,20,65,20,25,30,10,20,25,15,10,10,10,4,4,2,1]  # 物品所占容量
values= [220,208,198,192,180,180,165,162,160,158,155,130,125,122,120,118,115,110,105,101,100,100,98,96,95,90,88,82,80,77,75,73,72,70,69,66,65,63,60,58,56,50,30,20,15,10,8,5,3,1]  # 物品价值
gene_num=50  # 染色体内基因的个数，在本例中即为物品个数
p_point_mutation = 0.3  # 突变频率
iter_times = 100  # 种群迭代次数
population_size = 100  # 种群内包含多少个个体
offspring_size = 30  # 每次产生多少后代

start = time.time()
# 初始化
evaluator = Evaluator(gene_num,values,weights,weight_MAX)  # 评价类
population = Population(population_size,evaluator)  # 种群

for i in range(iter_times):
    selector_operator = SelectionOperator(population,offspring_size)  # 选择算子
    parents = selector_operator.roulette_wheel()  # 使用轮盘赌算法选择父代

    variation_operator = VariationOperator(evaluator,parents,p_point_mutation)  # 变异算子（交叉、突变）
    offspring = variation_operator.get_offspring()  # 经交叉和突变后获得子代
    
    population.evolve(offspring)  # 子代加入种群，种群内进行淘汰，维持原始个体数

# 结果展示
best_individual = max(population.best_individual_dict, key=population.best_individual_dict.get)
print('Time used: ', time.time()-start, 's')
print('best chromosome: ',best_individual.chromosome)
print('best fitness value: ',best_individual.fitness_value)
```

输出：

```python
Time used:  1.8347599506378174 s
best chromosome:  [1 1 1 1 0 1 1 1 0 1 1 0 0 0 1 1 0 0 1 1 1 0 0 1 0 0 1 1 1 1 0 0 0 0 1 0 1
 1 0 1 1 1 0 1 1 1 1 1 1 1]
best fitness value:  3028
```

#### 定义评价类

评价类内共2个方法:

* get_fitness_value：计算个体的**适应度**（目标函数）
* if_valid_chromosome：判断染色体是否有效（是否为可行解）

```python
class Evaluator(object):
    def __init__(self,gene_num,values,weights, weight_MAX):  
        self.gene_num = gene_num  # 染色体中基因的数量，在本问题中即为物品的个数
        self.values = values  # 各物品对应的价值
        self.weights = weights  # 各物体所占的容量
        self.weight_MAX = weight_MAX  # 背包的容量上限
    
    def get_fitness_value(self, individual):
        individual.fitness_value = sum(individual.chromosome*self.values)
        return individual.fitness_value
    
    def if_valid_chromosome(self,chromosome):
        return sum(chromosome*self.weights) > self.weight_MAX
```

#### 定义个体

个体类中共两个方法：

* get_chromosome：构建个体的染色体，可随机构建（用于初始化），也可指定（产生子代时需要指定）
* encode_chromosome：随机构建染色体，在本例中则随机构建长度为50的二进制列

```python
class Individual(object):
    def __init__(self,evaluator,fixed_chromosome=[]):
        self.evaluator = evaluator
        self.gene_num = evaluator.gene_num # 基因数量
        self.chromosome = self.get_chromosome(fixed_chromosome)  # 染色体       
        
    def get_chromosome(self,fixed_chromosome=[]):
        if fixed_chromosome:
            chromosome = np.array(fixed_chromosome)
        else:
            chromosome = self.encode_chromosome()
        while self.evaluator.if_valid_chromosome(chromosome):  # 判断染色体是否合法
            chromosome = np.random.randint(0,2,self.gene_num)
        return chromosome

    def encode_chromosome(self):
        return np.random.randint(0,2,self.gene_num)
```

#### 定义种群

种群中共1个方法:

* evolve：种群进化，获得子代后在父代子代中根据适应度进行淘汰，维持种群内个体的数量。

```python
class Population(object):
    def __init__(self, population_size, evaluator, fixed_chromosome=[]):
        self.population_size = population_size  # 种群中个体数量
        self.evaluator = evaluator  # 评价类
        self.individuals = []  # 种群内的个体集合
        for n in range(population_size):
            self.individuals.append(Individual(self.evaluator,fixed_chromosome=[]))
            
        self.best_individual_list = []  # 记录每一代中最好的个体
        self.best_individual_dict = {}  # 记录每一代中最好的个体及其适应度
        
    def evolve(self,offspring):
        fv_now_gen = []  # 当前代的种群中个体的适应度列表
        for individual in self.individuals:
            fv_now_gen.append(self.evaluator.get_fitness_value(individual))
        
        best_index = fv_now_gen.index(max(fv_now_gen))
        self.best_individual_list.append(self.individuals[best_index])  # 记录当前代中的最优个体
        self.best_individual_dict[self.individuals[best_index]] = max(fv_now_gen)  # 记录当前代中最优个体及其适应度

        # 加入子代进行淘汰，维持种群内个体个数
        fv_offspring=[]  # 子代的种群中个体的适应度列表
        for individual in offspring:
            fv_offspring.append(self.evaluator.get_fitness_value(individual))
        
        # 本例中种群内个体个数为100，则对适应度进行排序后，查找第100个适应度的值作为分界值，
        # 筛选适应度大于等于分界线的个体
        fv_all = fv_now_gen+fv_offspring  # 父代+子代的适应度列表
        fv_all.sort(reverse=True)
        cut_value = fv_all[self.population_size-1]  # 查找分界值
        
        new_population = []
        for individual in self.individuals:
            if self.evaluator.get_fitness_value(individual) >= cut_value:
                new_population.append(individual)
        for individual in offspring:
            if self.evaluator.get_fitness_value(individual) >= cut_value:
                new_population.append(individual)
                
        self.individuals = new_population  # 更新种群内的个体，进入下一代

```

#### 定义选择算子类

选择算子有很多种，其中轮盘赌算法是最常见的一种。

轮盘赌算法的核心思想：

1） 适应度转化为概率：Pi = FitnessValue_i/sum(FitnessValue_all)。

2） 计算累积概率，将所有个体出现的可能转化到0-1上,适应度越大的个体，在0-1上的占比越大。

3） 随机在0-1中取数，根据随机数落在0-1的范围选择对应的个体，理论上适应度越高的个体被选中的概率越大。

选择算子类中共1种方法：

* roulette_wheel：轮盘赌算法

```python
class SelectionOperator(object):
    def __init__(self,population,parent_size):
        self.population = population  # 种群
        self.parent_size = parent_size  # 进入繁殖的父代个体数
        
    def roulette_wheel(self):  # 轮盘赌算法
        fitness_value_list=[]
        for individual in self.population.individuals:
            fitness_value_list.append(self.population.evaluator.get_fitness_value(individual))
        
        # 计算累计概率
        cumulate_p_list = []
        pre_p = 0
        for fitness_value in fitness_value_list:
            p=float(fitness_value/sum(fitness_value_list))
            cumulate_p_list.append(p+pre_p)
            pre_p = p+pre_p
        
        # 轮盘赌算法，适应度越大的个体越容易被选做可以繁殖的父代
        self.parents = []
        for n in range(self.parent_size):
            r = random.uniform(0,1)
            parents_pair=[]
            #print(n,'*'*10)
            for parent_num in range(2):  # 2个父代产生1个子代
                for i in range(len(cumulate_p_list)):
                    if i == 0:
                        if r <= cumulate_p_list[i]:
                            parents_pair.append(self.population.individuals[i])
                    else:
                        if r <= cumulate_p_list[i] and r > cumulate_p_list[i-1]:
                            parents_pair.append(self.population.individuals[i])
            self.parents.append(tuple(parents_pair))

        return self.parents
```

#### 定义变异算子类

变异算子类中包含3个方法：

* get_offspring：获得子代
* crossover：交叉
* point_mutation：点突变

```python
class VariationOperator(object):
    def __init__(self,evaluator,parents,p_point_mutation):
        self.evaluator = evaluator  # 评价类
        self.parents = parents  # 父代
        self.p_point_mutation = p_point_mutation  # 点突变概率
        self.offspring= []
        
    def get_offspring(self):
        self.crossover()  # 父代染色体交叉产生子代染色体
        self.point_mutation()  # 点突变
        return self.offspring
    
    def crossover(self):
        for pair in parents:
            cross_point = random.randint(0, evaluator.gene_num-1)  # 随机选取交叉点
            a_offspring_chromosome = []
            a_offspring_chromosome.extend(pair[0].chromosome[:cross_point])
            a_offspring_chromosome.extend(pair[1].chromosome[cross_point:])
            
            # 判断染色体是否合法
            if not self.evaluator.if_valid_chromosome(np.array(a_offspring_chromosome)):
                individual = Individual(self.evaluator,fixed_chromosome=a_offspring_chromosome)
                self.offspring.append(individual)
    
    def point_mutation(self):
        for i in range(len(self.offspring)):
            tmp = random.uniform(0,1)  
            if tmp < p_point_mutation:  # 突变概率
                mutation = random.randint(0, evaluator.gene_num-1)  # 随机选取突变点
                self.offspring[i].chromosome[mutation] = bool(1-self.offspring[i].chromosome[mutation])  # 点突变
```

