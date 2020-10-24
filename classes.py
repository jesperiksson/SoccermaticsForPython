#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:00:30 2020

@author: jesper
"""
import pandas as pd
class Table():
    # Makes a table 
    def __init__(self):
        self.table = pd.DataFrame(
            columns = ['Team','Points','Win','Draw','Lose','Goals for','Goals against','Goal difference']
            )
    def add_numbers(self,team_list):
        for i in range(len(team_list)):
            t = team_list[i]
            self.table = self.table.append(
                pd.DataFrame(
                    [[t.name,(t.wins*3+t.draws*1),t.wins,t.draws,t.losses,
                      t.goals_for,t.goals_against,(t.goals_for-t.goals_against)]],
                    columns= ['Team','Points','Win','Draw','Lose','Goals for','Goals against','Goal difference']
                    )
                )
        self.table = self.table.sort_values(by='Points',ascending=False)
        self.table.index = range(1,len(self.table)+1)
    
    def show_table(self):
        return self.table
    
class Team():
    # Team objects which populate the Table
    def __init__(self,name):
        self.name = name
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.goals_for = 0
        self.goals_against = 0
        
    def add_result(self,scored,conceded):
        if scored > conceded:
            self.wins += 1
        elif scored == conceded:
            self.draws += 1
        else:
            self.losses +=1
        self.goals_for += scored
        self.goals_against += conceded