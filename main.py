#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:54:26 2020

@author: jesper
"""

import pandas as pd

from classes import Stats, Simulation

# Make DataFrames
df1920 = pd.read_csv('PL1920.csv')
df1819 = pd.read_csv('PL1819.csv')



#%% Simulate a single season
S = Stats(df1920)
t = S.simulate_season()
s = t.show_table()
print(s)
#%% CASE 1 - Naive approach: mu = (team A attack + team B defence) / 2. Set order of games 
sim1 = Simulation(df1920,1000)

sim1.simulate_seasons()

sim1.calc_freq()
sim1.plot_position_crosstab()

