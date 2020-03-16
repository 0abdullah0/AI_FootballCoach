from flask import Flask, jsonify, request
import pandas as pd
import numpy  as np
import seaborn
import copy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson,skellam
import goals

app = Flask(__name__)

def Poisson_model(dataset,home,away):
    goal_model_data = pd.concat([dataset[['HomeTeam', 'AwayTeam', 'HomeGoals']].assign(home=1).rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'HomeGoals': 'goals'}),
        dataset[['AwayTeam', 'HomeTeam', 'AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'AwayGoals': 'goals'})])

    poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                            family=sm.families.Poisson()).fit()
    poisson_model.summary()
    return poisson_model

def simulate_match(foot_model, homeTeam, awayTeam, max_goals=8):
        home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                               'opponent': awayTeam, 'home': 1},
                                                         index=[1])).values[0]
        away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                               'opponent': homeTeam, 'home': 0},
                                                         index=[1])).values[0]
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                     [home_goals_avg, away_goals_avg]]
        return (np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

def build_teams():
    teams=[]
    season0910 = pd.read_csv("Prediction_Dataset/season-0910_csv.csv")
    season0910 = season0910[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season0910 = season0910.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1011 = pd.read_csv("Prediction_Dataset/season-1011_csv.csv")
    season1011 = season1011[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1011 = season1011.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1112 = pd.read_csv("Prediction_Dataset/season-1112_csv.csv")
    season1112 = season1112[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1112 = season1112.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1213 = pd.read_csv("Prediction_Dataset/season-1213_csv.csv")
    season1213 = season1213[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1213 = season1213.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1314 = pd.read_csv("Prediction_Dataset/season-1314_csv.csv")
    season1314 = season1314[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1314 = season1314.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1415 = pd.read_csv("Prediction_Dataset/season-1415_csv.csv")
    season1415 = season1415[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1415 = season1415.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1516 = pd.read_csv("Prediction_Dataset/season-1516_csv.csv")
    season1516 = season1516[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1516 = season1516.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1617 = pd.read_csv("Prediction_Dataset/season-1617_csv.csv")
    season1617 = season1617[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1617 = season1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1718 = pd.read_csv("Prediction_Dataset/season-1718_csv.csv")
    season1718 = season1718[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1718 = season1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1819 = pd.read_csv("Prediction_Dataset/season-1819_csv.csv")
    season1819 = season1819[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1819 = season1819.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    frames = [season0910,season1011,season1112,season1213,season1314,season1415,season1516,season1617,season1718,season1819]
    result = pd.concat(frames)

    teams=result['HomeTeam']
    teams=list(dict.fromkeys(teams))
    return teams

def get_history(team):

    season0910 = pd.read_csv("Prediction_Dataset/season-0910_csv.csv")
    season0910 = season0910[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season0910 = season0910.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1011 = pd.read_csv("Prediction_Dataset/season-1011_csv.csv")
    season1011 = season1011[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1011 = season1011.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1112 = pd.read_csv("Prediction_Dataset/season-1112_csv.csv")
    season1112 = season1112[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1112 = season1112.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1213 = pd.read_csv("Prediction_Dataset/season-1213_csv.csv")
    season1213 = season1213[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1213 = season1213.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1314 = pd.read_csv("Prediction_Dataset/season-1314_csv.csv")
    season1314 = season1314[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1314 = season1314.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1415 = pd.read_csv("Prediction_Dataset/season-1415_csv.csv")
    season1415 = season1415[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1415 = season1415.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1516 = pd.read_csv("Prediction_Dataset/season-1516_csv.csv")
    season1516 = season1516[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1516 = season1516.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1617 = pd.read_csv("Prediction_Dataset/season-1617_csv.csv")
    season1617 = season1617[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1617 = season1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    season1718 = pd.read_csv("Prediction_Dataset/season-1718_csv.csv")
    season1718 = season1718[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1718 = season1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})


    season1819 = pd.read_csv("Prediction_Dataset/season-1819_csv.csv")
    season1819 = season1819[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']]
    season1819 = season1819.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','FTR': 'Results'})

    frames = [season1112,season1213,season1314,season1415,season1516,season1617,season1718]
    result = pd.concat(frames)

    home_hist = result[result.get('HomeTeam') == team]
    away_hist = result[result.get('AwayTeam') == team]
    return result,home_hist ,away_hist

def most_goals_win(home,away,draw):

    maxHome=goals.Goals(-1,-1,-1)
    maxAway=goals.Goals(-1,-1,-1)
    maxDraw=goals.Goals(-1,-1,-1)

    for i in home:
        if(i.prob>maxHome.prob):
            maxHome=copy.deepcopy(i)
    for i in away:
        if(i.prob>maxAway.prob):
            maxAway=copy.deepcopy(i)
    for i in draw:
        if(i.prob>maxDraw.prob):
            maxDraw=copy.deepcopy(i)
    return maxHome,maxAway,maxDraw

def prob_report(expected):
    home_win=0
    home_goal=[]

    draw=0
    draw_goal = []

    away_win=0
    away_goal = []

    for i in range(len(expected)):
        for j in range(len(expected[i])):
            if(i>j):
                home_win+=expected[i][j]
                home_goal.append(goals.Goals(i,j,expected[i][j]))
            elif(j>i):
                away_win += expected[i][j]
                away_goal.append(goals.Goals(i,j,expected[i][j]))
            else:
                draw+=expected[i][j]
                draw_goal.append(goals.Goals(i,j,expected[i][j]))

    hgoal,agoal,dgoal=most_goals_win(home_goal,away_goal,draw_goal)


    print("Away Win:", round(away_win,2)*100,"% with",agoal.Home,"-",agoal.Away)
    print("Darw:" , round(draw,2)*100,"% with",dgoal.Home,"-",dgoal.Away)

    return home_win,hgoal.Home,hgoal.Away,away_win,agoal.Home,agoal.Away,draw,dgoal.Home,dgoal.Away



@app.route('/predict', methods=['GET'])
def predict():
    home=request.args.get('home')
    away = request.args.get('away')
    teams = []
    teams = build_teams()
    for i in range(len(teams)):
        dataset, homeHist, awayHist = get_history(teams[i])
    p = Poisson_model(dataset, home, away)
    expected = simulate_match(p, home, away, max_goals=10)
    hwin,hgh,hga,awin,agh,aga,draw,dgh,dga=prob_report(expected)
    return jsonify({'Home Win':
                         {'Percentage': str(round(hwin, 2) * 100),
                          'Homegoals': str(hgh),
                          'Awaygoals': str(hga)}
                     },
                    {'Away Win':
                         {'Percentage': str(round(awin, 2) * 100),
                          'Homegoals': str(agh),
                          'Awaygoals': str(aga)}
                     },
                    {'Draw':
                         {'Percentage': str(round(draw, 2) * 100),
                          'Homegoals': str(dgh),
                          'Awaygoals': str(dga)}
                     }
                   )

if __name__ == '__main__':
    app.run(debug=True)