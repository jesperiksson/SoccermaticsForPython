#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:54:26 2020

@author: jesper
"""

import pandas as pd
import numpy as np

# Make DataFrames
df1920 = pd.read_csv('PL1920.csv')
df1819 = pd.read_csv('PL1920.csv')

def simulate_game_poisson(home_expected_scored, home_expected_conceded, away_expected_scored, away_expected_conceded):
    # Simple model to predict the result using poisson distribution
    home_expected = (home_expected_scored + away_expected_conceded)/2
    away_expected = (away_expected_scored + home_expected_conceded)/2
    home_goals = np.random.poisson(home_expected)
    away_goals = np.random.poisson(away_expected)    
    return home_goals, away_goals

def get_expected_values(df):
    teams =list(set(df['HomeTeam']))
    expected_values = pd.DataFrame(columns = ['Team','ExpectedScored','ExpectedConceded'])
    for i in range(len(teams)):
        avg_score = (np.sum(df.loc[df['HomeTeam'] == teams[i]]['FTHG']) + np.sum(df.loc[df['AwayTeam'] == teams[i]]['FTAG']))/(len(df)/len(teams)*2)
        avg_letin = (np.sum(df.loc[df['HomeTeam'] == teams[i]]['FTAG']) + np.sum(df.loc[df['AwayTeam'] == teams[i]]['FTHG']))/(len(df)/len(teams)*2)
        expected_values = expected_values.append(
            pd.DataFrame(
                [[teams[i],avg_score,avg_letin]], columns= ['Team','ExpectedScored','ExpectedConceded']
                )
            )
    expected_values.index = range(1,len(teams)+1)
    return expected_values

def simulate_season(df):
    home_teams = list(df['HomeTeam'])
    away_teams = list(df['AwayTeam'])
    
    input_values = get_expected_values(df)
    teams = list(input_values['Team'])
    team_dict = {}
    for i in range(len(teams)):
        team_dict.update({
            teams[i] : Team(teams[i])})
    for i in range(len(df)):
        home_team = home_teams[i]
        away_team = away_teams[i]
        home_goals, away_goals = simulate_game_poisson(
            float(input_values.loc[input_values['Team'] == home_team]['ExpectedScored']),
            float(input_values.loc[input_values['Team'] == home_team]['ExpectedConceded']),
            float(input_values.loc[input_values['Team'] == away_team]['ExpectedScored']),
            float(input_values.loc[input_values['Team'] == away_team]['ExpectedConceded'])
            )
        team_dict[home_team].add_result(home_goals,away_goals)
        team_dict[away_team].add_result(away_goals,home_goals)
    
    table = Table()
    table.add_numbers(list(team_dict.values()))
     
    return table

class Table():
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
    
    def show_table(self):
        sorted_table = self.table.sort_values(by='Points',ascending=False)
        sorted_table.index = range(1,len(sorted_table)+1)
        return sorted_table
    
class Team():
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
        
#%%
t = simulate_season(df1920)
s = t.show_table()
print(s)