#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:35:12 2020

@author: BenjaminMeco
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from pandas import json_normalize
from FCPython import createPitch
import statsmodels.formula.api as smf

def factorial(n):
    if(n == 0):
        return 1
    else:
        return n*factorial(n-1)

def pois(l,k):
    return (l**k)*np.exp(-l)/factorial(k)

# this is just a help for getting the data
def indexOf(team_name,team_list):
    index = -1
    for element in team_list:
        index = index + 1
        if(element[0] == team_name):
            return index
    
    return -1

# for getting the distributions:

def getPoisson(number):

    result = np.zeros(11)
    for k in range(0,11):
        result[k] = pois(number,k)
    return result

def getWeights(arr,size):
    weights = np.zeros(size)
    W = 0
    for k in range(0,size):
        W = W + arr[k]
    for k in range(0,size):
        weights[k] = arr[k]/W
    return weights

def outcomeWeights(r,probabilities):
    s = probabilities[0]
    count = 0
    for p in probabilities:
        if(s > r):
            return count
        else:
            count = count + 1
            s = s + p
    return count
    

# this makes a simulation using weights as the probability distribution
def simulate(team_list):
    points = np.zeros(len(team_list))
    
    for i in range(0,len(team_list)):
        for j in range(1,len(team_list)):
            t_1 = team_list[i]
            t_2 = team_list[(i+j)%len(team_list)]
            g_1 = outcomeWeights(np.random.uniform(0,1),t_1[1])
            c_1 = outcomeWeights(np.random.uniform(0,1),t_1[2])
            g_2 = outcomeWeights(np.random.uniform(0,1),t_2[1])
            c_2 = outcomeWeights(np.random.uniform(0,1),t_2[2])
            if(g_1 + c_2 > g_2 + c_1):
                points[i] = points[i] + 3
            elif(g_1 + c_2 < g_2 + c_1):
                points[(i+j)%len(team_list)] = points[(i+j)%len(team_list)] + 3
            else:
                points[i] = points[i] + 1
                points[(i+j)%len(team_list)] = points[(i+j)%len(team_list)] + 1
    
    result = []
    for i in range(0,len(team_list)):
        result = result + [[points[i],team_list[i][0]]]
    return result
                

def simulMany(team_list,N):

    team_placements = []
    for t in team_list:
        team_placements = team_placements + [[t[0],np.zeros(21)]]
    
    for n in range(N):
        # do a simulation:
        s = sorted(simulate(team_list))
        
        # get the placements:
        for i in range(0,len(s)):
            e = s[i]
            index = indexOf(e[1],team_list)
            team_placements[index][1][20-i] = team_placements[index][1][20-i] + 1
    
    for t in team_placements:
        t[1] = getWeights(t[1],21)[1:]
    
    return team_placements

#Load the data
with open('Wyscout/matches/matches_England.json') as data_file:
    data = json.load(data_file)

df = json_normalize(data, sep = "_")

# first we extract the relevant bits of the matches:

matches = []
for i,game in df.iterrows():
    label = game['label']
    dash = label.find(" -")
    comma = label.find(",")
    team_1 = label[0:dash]
    team_2 = label[dash+3:comma]
    score_1 = label[comma+2:comma+3]
    score_2 = label[comma+6:]
    matches = matches + [[team_1,score_1,team_2,score_2]]

# now we make the distributions for each team:
teamList = []

for m in matches:
    index_1 = indexOf(m[0],teamList)
    index_2 = indexOf(m[2],teamList)

    # update the data for the first team
    if(index_1 == -1):
        new_team = [m[0],0,0]
        new_team[1] = int(m[1])
        new_team[1] = int(m[3])
        teamList = teamList + [new_team]
    else:
        teamList[index_1][1] = teamList[index_1][1] + int(m[1])
        teamList[index_1][2] = teamList[index_1][2] + int(m[3])
        
    # update the data for the second team
    if(index_2 == -1):
        new_team = [m[2],0,0]
        new_team[1] = int(m[3]) 
        new_team[2] = int(m[1])
        teamList = teamList + [new_team]
    else:
        teamList[index_2][1] = teamList[index_2][1] + int(m[3])
        teamList[index_2][2] = teamList[index_2][2] + int(m[1])

teamList.sort()

# now we get the desired data for the weights and the poisson distributions:
teamPoisson = []
    
for t in teamList:
    teamPoisson = teamPoisson + [[t[0],getPoisson(t[1]/38),getPoisson(t[2]/38)]]

# finally some simulations, first with the Poisson distributions:
N = 100
team_placements = simulMany(teamPoisson,N)

'''
for t in team_placements:

    plt.plot(range(1,21),t[1], color = "blue")
    plt.xlabel("Placement")
    plt.ylabel("Probability of placement for " + t[0])
    plt.xticks(range(1,21))
    plt.show()
'''

# next we look at how the performance of liverpool changes when they 
# improve the offence/defence or both. We do this by changing their parameters in the
# poisson distribution.  

lambda_off = teamList[indexOf("Liverpool",teamList)][1]/38
lambda_def = teamList[indexOf("Liverpool",teamList)][2]/38
plt.figure(dpi = 160)

# first we look at improving offence:
for d in np.linspace(10,25,2):
    print(str(d))
    # make the modifications:: 
    for k in range(0,11):
        teamPoisson[indexOf("Liverpool",teamPoisson)][1][k] = pois(lambda_off + d/38,k)
        teamPoisson[indexOf("Liverpool",teamPoisson)][2][k] = pois(lambda_def,k)
    # simulate and plot the distributions of Liverpool:
    N = 100
    T = simulMany(teamPoisson,N)
    t = T[indexOf("Liverpool",T)]
    plt.plot(range(1,21),t[1], color = (0.2+0.8*d/25,0,0),label = "Scoring " +str(d) + " more goals")

# secondly we look at improving defence:
for d in np.linspace(10,25,2):
    print(str(d))
    # make the modifications:: 
    for k in range(0,11):
        teamPoisson[indexOf("Liverpool",teamPoisson)][1][k] = pois(lambda_off,k)
        teamPoisson[indexOf("Liverpool",teamPoisson)][2][k] = pois(lambda_def-d/38,k)
    
    # simulate and plot the distributions of Liverpool:
    N = 100
    T = simulMany(teamPoisson,N)
    t = T[indexOf("Liverpool",T)]
    plt.plot(range(1,21),t[1], color = (0,0,0.2+0.8*d/25),label = "Conceding " +str(d) + " fewer goals")

plt.plot(range(1,21),team_placements[indexOf("Liverpool",team_placements)][1],color = "black",label = "Baseline")
plt.xlabel("Placement")
plt.ylabel("Probability of placement for Liverpool\n with improvements")
plt.xticks(range(1,21))
plt.xlim(0,10)
plt.legend()
plt.show()

    