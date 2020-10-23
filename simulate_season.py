#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:54:26 2020

@author: jesper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Make DataFrames
df1920 = pd.read_csv('PL1920.csv')
df1819 = pd.read_csv('PL1819.csv')

#https://towardsdatascience.com/visualizing-the-2019-20-english-premier-league-season-with-matplotlib-and-pandas-fd491a07cfda    
team_colors = {'Arsenal':'#ef0107', 'Aston Villa':'#95bfe5', 'Bournemouth':'#da291c', 'Brighton':'#0057b8',
               'Burnley':'#6c1d45', 'Chelsea':'#034694', 'Crystal Palace':'#1b458f', 'Everton':'#003399',
               'Leicester':'#003090', 'Liverpool':'#c8102e', 'Man City':'#6cabdd', 'Man United':'#da291c',
               'Newcastle':'#241f20', 'Norwich':'#fff200', 'Sheffield United':'#ee2737', 
               'Southampton':'#d71920', 'Tottenham':'#132257', 'Watford':'#fbee23', 'West Ham':'#7a263a',
               'Wolves':'#fdb913'}


def simulate_game_poisson(home_expected_scored, home_expected_conceded, away_expected_scored, away_expected_conceded):
    # Simple model to predict the result using poisson distribution
    home_expected = (home_expected_scored + away_expected_conceded)/2
    away_expected = (away_expected_scored + home_expected_conceded)/2
    home_goals = np.random.poisson(home_expected)
    away_goals = np.random.poisson(away_expected)    
    return home_goals, away_goals


def get_expected_values(df):
    # Generates a DataFrame with teams and their excpected values
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
    # Main funciton
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
        

#%% Simulate a singloe season
t = simulate_season(df1920)
s = t.show_table()
print(s)
#%% CASE 1 - Naive approach: mu = (team A attack + team B defence) / 2. Set order of games 
seasons = 300
season_list = np.array([])
for i in range(seasons):
    season_list = np.append(season_list,simulate_season(df1920))
#%% Plot hist with selected teams positions
fig = plt.figure(dpi=400)
ax = fig.add_subplot(111)
teams = list(team_colors.keys())
teams = ['Man City','Liverpool','Arsenal','Chelsea','Man United']
places = np.array([])
for team in teams:
    try:
        places = np.concatenate((places,
            [season_list[season].table.loc[season_list[season].table['Team'] == team].index for season in range(len(season_list))]),
            axis = -1)
    except ValueError:
        places = np.array(
            [season_list[season].table.loc[season_list[season].table['Team'] == team].index for season in range(len(season_list))])

average_place = np.round(np.average(places),decimals = 0)
median_place = np.median(places)
plt.hist(places, 
         bins = np.arange(1, places.max() + 1.5) - 0.5, 
         histtype = 'bar',
         color = [team_colors[t] for t in teams],
         ec = 'k',
         alpha = 0.9,
         zorder = 2)
plt.xticks(range(int(np.min(places)),int(np.max(places)+1)))
plt.xlabel('Position')
plt.ylabel('Frequency')
plt.title('End of season placement distribution over ' + str(seasons)+' seasons')
plt.legend(teams)
ax.set_facecolor('lightgray')
ax.grid(color = 'white',linewidth = 0.2,zorder = 1)
plt.show()
#%% Calculate frequency of each final position for each team
rows = 4
cols = 5
n_places = cols*rows
teams = list(team_colors.keys())


places = np.array([])
for team in teams:
    try:
        places = np.concatenate((places,
            [season_list[season].table.loc[season_list[season].table['Team'] == team].index for season in range(len(season_list))]),
            axis = -1)
    except ValueError:
        places = np.array(
            [season_list[season].table.loc[season_list[season].table['Team'] == team].index for season in range(len(season_list))])

freq = np.array([sum(places[:,t]==place+1) for t in range(len(teams)) for place in range(n_places)])
freq = freq.reshape((len(teams),n_places)) # rows : teams, cols : places
#%% Draw a pie chart for the distribution for each place

for i in range(n_places):
    plt.figure(dpi=400)
    #ax = fig.add_subplot(rows,cols,i+1)
    plt.pie(
        freq[:,i],
        labels = list(team_colors.keys()),
        colors = list(team_colors.values()),
        autopct='%1.1f%%',  # Draw percentage
        rotatelabels=True)
    plt.show()