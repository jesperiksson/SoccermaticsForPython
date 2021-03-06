#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:00:30 2020

@author: jesper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sn

class Table():
    # Makes a table at the end of a season
    def __init__(self):
        self.table = pd.DataFrame( # Initiate an empty table
            columns = ['Team','Points','Win','Draw','Lose','Goals for','Goals against','Goal difference']
            )
    def add_numbers(self,team_list): # Is called from the simulate_season method of the Stats class
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
        if scored > conceded: # win
            self.wins += 1
        elif scored == conceded: # draw
            self.draws += 1
        else: # loss
            self.losses +=1 
        self.goals_for += scored
        self.goals_against += conceded

class Stats(): # raw data, teams, colors, parameters, etc. 
    def __init__(self,df):
        self.df = df
        self.team_colors = {'Arsenal':'#ef0107', 'Aston Villa':'#95bfe5', 'Bournemouth':'#da291c', 'Brighton':'#0057b8',
               'Burnley':'#6c1d45', 'Chelsea':'#034694', 'Crystal Palace':'#1b458f', 'Everton':'#003399',
               'Leicester':'#003090', 'Liverpool':'#c8102e', 'Man City':'#6cabdd', 'Man United':'#da291c',
               'Newcastle':'#241f20', 'Norwich':'#fff200', 'Sheffield United':'#ee2737', 
               'Southampton':'#d71920', 'Tottenham':'#132257', 'Watford':'#fbee23', 'West Ham':'#7a263a',
               'Wolves':'#fdb913'}
#https://towardsdatascience.com/visualizing-the-2019-20-english-premier-league-season-with-matplotlib-and-pandas-fd491a07cfda    
        
        self.teams =list(set(df['HomeTeam']))        
        self.home_teams = list(df['HomeTeam'])
        self.away_teams = list(df['AwayTeam'])
        expected_values = pd.DataFrame(columns = ['Team','ExpectedScored','ExpectedConceded']) # initiate empty DataFrame
        # Naive approach, each team has a offense and a defense expected value
        # Generates a DataFrame with teams and their excpected values
        for i in range(len(self.teams)):
            avg_score = (np.sum(df.loc[df['HomeTeam'] == self.teams[i]]['FTHG']) + np.sum(df.loc[df['AwayTeam'] == self.teams[i]]['FTAG']))/(len(df)/len(self.teams)*2)
            avg_letin = (np.sum(df.loc[df['HomeTeam'] == self.teams[i]]['FTAG']) + np.sum(df.loc[df['AwayTeam'] == self.teams[i]]['FTHG']))/(len(df)/len(self.teams)*2)
            expected_values = expected_values.append( # Populate the DataFrame
                pd.DataFrame(
                    [[self.teams[i],avg_score,avg_letin]], columns= ['Team','ExpectedScored','ExpectedConceded']
                    )
                )
        expected_values.index = range(1,len(self.teams)+1)
        self.expected_values = expected_values # The input values for the naive approach
        
        # Including home advantage, each team has two home and away parameters
        # Generates a DataFrame with teams and their excpected values
        expected_values_home = pd.DataFrame(columns = ['Team','ExpectedScored','ExpectedConceded'])
        expected_values_away = pd.DataFrame(columns = ['Team','ExpectedScored','ExpectedConceded'])
        for i in range(len(self.teams)):
            avg_score_home = (np.sum(df.loc[df['HomeTeam'] == self.teams[i]]['FTHG']))/(len(df)/len(self.teams))
            avg_letin_home = (np.sum(df.loc[df['HomeTeam'] == self.teams[i]]['FTAG']))/(len(df)/len(self.teams))
            avg_score_away = (np.sum(df.loc[df['AwayTeam'] == self.teams[i]]['FTAG']))/(len(df)/len(self.teams))
            avg_letin_away = (np.sum(df.loc[df['AwayTeam'] == self.teams[i]]['FTHG']))/(len(df)/len(self.teams))
            expected_values_home = expected_values_home.append(
                pd.DataFrame(
                    [[self.teams[i],avg_score_home,avg_letin_home]], columns = ['Team','ExpectedScored','ExpectedConceded'])
                )
            expected_values_away = expected_values_away.append(
                pd.DataFrame(
                    [[self.teams[i],avg_score_away,avg_letin_away]], columns = ['Team','ExpectedScored','ExpectedConceded'])
                )
        expected_values_home.index = range(1,len(self.teams)+1)
        expected_values_away.index = range(1,len(self.teams)+1)
        self.expected_values_home = expected_values_home # The input values when 
        self.expected_values_away = expected_values_away # considering home advantage

    def simulate_game_poisson(self,home_expected_scored, home_expected_conceded, away_expected_scored, away_expected_conceded):
        # Simple model to predict the result using poisson distribution
        home_expected = (home_expected_scored + away_expected_conceded)/2
        away_expected = (away_expected_scored + home_expected_conceded)/2
        home_goals = np.random.poisson(home_expected)
        away_goals = np.random.poisson(away_expected)    
        return home_goals, away_goals
    

    
    def simulate_season(self):
        # A single season
        team_dict = {} # Using a dictionary to keep track of the Team instances
        for i in range(len(self.teams)):
            team_dict.update({
                self.teams[i] : Team(self.teams[i])})
        for i in range(len(self.df)):
            home_team = self.home_teams[i]
            away_team = self.away_teams[i]
            home_goals, away_goals = self.simulate_game_poisson(
                float(self.expected_values.loc[self.expected_values['Team'] == home_team]['ExpectedScored']),
                float(self.expected_values.loc[self.expected_values['Team'] == home_team]['ExpectedConceded']),
                float(self.expected_values.loc[self.expected_values['Team'] == away_team]['ExpectedScored']),
                float(self.expected_values.loc[self.expected_values['Team'] == away_team]['ExpectedConceded'])
                )
            team_dict[home_team].add_result(home_goals,away_goals)
            team_dict[away_team].add_result(away_goals,home_goals)       
        table = Table()
        table.add_numbers(list(team_dict.values()))
        return table
    
    def simulate_season_homeaway(self): # Same as above but considering home advantage and away disadvantage
        team_dict = {}
        for i in range(len(self.teams)):
            team_dict.update({
                self.teams[i] : Team(self.teams[i])})
        for i in range(len(self.df)):
            home_team = self.home_teams[i]
            away_team = self.away_teams[i]
            home_goals, away_goals = self.simulate_game_poisson(
                float(self.expected_values_home.loc[self.expected_values_home['Team'] == home_team]['ExpectedScored']),
                float(self.expected_values_home.loc[self.expected_values_home['Team'] == home_team]['ExpectedConceded']),
                float(self.expected_values_away.loc[self.expected_values_away['Team'] == away_team]['ExpectedScored']),
                float(self.expected_values_away.loc[self.expected_values_away['Team'] == away_team]['ExpectedConceded'])
                )
            team_dict[home_team].add_result(home_goals,away_goals)
            team_dict[away_team].add_result(away_goals,home_goals)    
        table = Table()
        table.add_numbers(list(team_dict.values())) 
        return table
            
    def poisson_regression(self):
        # TBI, using a regression model from some module and compare it to my results
        pass

class Simulation(Stats): # Same as Stats but for simulating several seasons
    def __init__(self,df,n,team_of_interest = 'Liverpool'):
        super().__init__(df)
        self.n_seasons = n # Number of seasons to simulate
        self.team_of_interest = team_of_interest
    
    def simulate_seasons(self):
        season_list = np.array([])
        for i in range(self.n_seasons): # building a list of all seasons
            season_list = np.append(season_list,self.simulate_season())
        self.season_list = season_list # Last season is stored 
            
    def simulate_seasons_homeaway(self):
        season_list = np.array([])
        for i in range(self.n_seasons): # building a list of all seasons
            season_list = np.append(season_list,self.simulate_season_homeaway())
        self.season_list_homeaway = season_list
    
    def plot_hist(self):
        #Plot histogram with selected teams positions
        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(111)
        cut_off = 6 # The lowest position for which to extend the histogram
        # teams = list(self.team_colors.keys()) # to use all teams
        teams = ['Man City','Liverpool','Arsenal','Chelsea','Man United'] # to use selected teams
        places = np.array([])
        for team in teams:
            try: # The main case
                places = np.concatenate((places, # store the positions for each team for each season
                    [self.season_list[season].table.loc[self.season_list[season].table['Team'] == team].index for season in range(len(self.season_list))]),
                    axis = -1)
            except ValueError: # accounting for the edge case where there is no season list
                places = np.array(
                    [self.season_list[season].table.loc[self.season_list[season].table['Team'] == team].index for season in range(len(self.season_list))])
        
        plt.hist(places, 
                 bins = np.arange(1, cut_off + 1.5) - 0.5, # Putting the bins over the xtick  
                 histtype = 'bar',
                 color = [self.team_colors[t] for t in teams], # colors of selected teams
                 ec = 'k', # Edgecolor 
                 alpha = 0.9, 
                 zorder = 2)
        plt.xticks(range(int(np.min(places)),int(np.max(places)+1)))
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        plt.title('End of season placement distribution over ' + str(len(self.season_list))+' seasons')
        plt.legend(teams)
        ax.set_facecolor('lightgray')
        ax.grid(color = 'white',linewidth = 0.2,zorder = 1)
        plt.show()
        
    def calc_freq(self):
        # Calculate frequency of each final position for each team
        try: # If there is a simulation with results that can be counted
            teams = self.teams
            places = np.array([])
            for team in teams:
                try: # This is redundant 
                    places = np.concatenate((places,
                        [self.season_list[season].table.loc[self.season_list[season].table['Team'] == team].index for season in range(len(self.season_list))]),
                        axis = -1)
                except ValueError:
                    places = np.array(
                        [self.season_list[season].table.loc[self.season_list[season].table['Team'] == team].index for season in range(len(self.season_list))])
            
            freq = np.array([sum(places[:,t]==place+1) for t in range(len(teams)) for place in range(len(teams))])
            freq = freq.reshape((len(teams),len(teams))) # rows : teams, cols : places
            self.freq = freq
        except NameError:
            print('No simulation has been ran')

    
    def plot_pie(self,rows = 4,cols = 5): # Draw a pie chart for the distribution for each place
        def my_autopct(pct): # Utility function for plot_pie
            return ('%.2f' % pct) if pct > 5 else ''
        n_places = cols*rows
        for i in range(n_places):
            plt.figure(dpi=400)
            #ax = fig.add_subplot(rows,cols,i+1)
            plt.pie(
                self.freq[:,i],
                explode = [0.1 if t == self.team_of_interest else 0 for t in self.teams],
                labels = list(self.team_colors.keys()),
                colors = list(self.team_colors.values()),
                autopct=my_autopct,  # Draw percentage
                labeldistance = 1,
                rotatelabels=True,
                radius=1.5)
            plt.tight_layout
            plt.title(
                'Proportion of the times each team ended up in position '+str(i+1),
                loc = 'center',
                pad = 100)
            plt.show()
            
    def plot_position_crosstab(self): # Plots the heatmap 
        fig = plt.figure(dpi=400)
        fig.set_size_inches(8,8)
        crosstab = pd.DataFrame(
            self.freq,
            columns=range(1,21),
            index = self.teams)
        crosstab = crosstab.sort_values(
            by=[crosstab.columns[i] for i in range(len(crosstab.columns))],
            ascending=[False]*len(crosstab.columns))
        self.crosstab = crosstab # If you want to do some statistics
        crosstab_percent = crosstab.div(np.ones((len(crosstab.columns),len(crosstab.index)))*self.n_seasons)
        crosstab_percent = crosstab_percent.mul(np.ones((len(crosstab.columns),len(crosstab.index)))*100)
        ax = sn.heatmap( # TBD, change the annotations to percentage of all seasons
            data=crosstab_percent.T,
            norm = colors.PowerNorm(gamma=0.4), #  Ḿore emphasis towards the green end
            cmap = 'RdYlGn', # Is there a better one?
            cbar = False,
            annot = True,
            fmt=".0f",
            xticklabels=True, 
            yticklabels=True)
        ax.set_yticks = range(1,21)
        #ax.tight_layout()
        ax.plot()