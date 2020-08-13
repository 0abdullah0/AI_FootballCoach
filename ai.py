from flask import Flask, jsonify, request,app
import pandas as pd
import numpy  as np
import copy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson,skellam
import goals
import seaborn
import matplotlib.pyplot as plt
from pandas._libs import json

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

    frames = [season0910,season1011,season1112,season1213,season1314,season1415,season1516,season1617,season1718,season1819]
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
    return jsonify({'HomeWin_Percentage': str(round(hwin,2) * 100),
                    'HomeWin_Homegoals' : str(hgh),
                    'HomeWin_Awaygoals' : str(hga),
                    'AwayWin_Percentage': str(round(awin,2) * 100),
                    'AwayWin_Homegoals': str(agh),
                    'AwayWin_Awaygoals': str(aga),
                    'Draw_Percentage': str(round(draw,2) * 100),
                    'Draw_Homegoals': str(dgh),
                    'Draw_Awaygoals': str(dga)
                     }
                   )

#_________________________________________________________________________________________________________
def read_Dataset():
    df = pd.read_csv("Recommend_Dataset/FIFA2019.csv")
    df.head(7)
    return df

def get_ClubPlayers(dataset,teamName):
    dataset=dataset[dataset.get('Club') == teamName]
    return dataset

def find_BestGoalkeeper(df):
    # weights
    a = 0.5
    b = 1
    c = 2
    d = 3

    # GoalKeeping Characterstics
    df['gk_Shot_Stopper']=(b * df.Reactions + b * df.Composure + a * df.Speed + a * df.Strength + c * df.Jumping + b * df.GK_Positioning + c * df.GK_Diving + d * df.GK_Reflexes + b * df.GK_Handling) / ( 2 * a + 4 * b + 2 * c + 1 * d)
    df['gk_Sweeper']=(b * df.Reactions + b * df.Composure + b * df.Speed + a * df.Short_Pass + a * df.Long_Pass + b * df.Jumping + b * df.GK_Positioning + b * df.GK_Diving + d * df.GK_Reflexes + b * df.GK_Handling + d * df.GK_Kicking + c * df.Vision) / (2 * a + 4 * b + 3 * c + 2 * d)

    gss= df[(df['Club_Position'] == 'GK')].sort_values('gk_Shot_Stopper', ascending=False)[:3]
    x1Names = np.array(list(gss['Name']))
    x1Photos = np.array(list(gss['Photo']))
    x1Numbers = np.array(list(gss['Club_Number']))

    gs = df[(df['Club_Position'] == 'GK')].sort_values('gk_Sweeper', ascending=False)[:3]
    x2Names = np.array(list(gs['Name']))
    x2Photos = np.array(list(gs['Photo']))
    x2Numbers = np.array(list(gs['Club_Number']))

    return x1Names,x1Photos,x1Numbers,x2Names,x2Photos,x2Numbers

def find_BestDefenders(df):
    a = 0.5
    b = 1
    c = 2
    d = 3

    # Choosing Defenders
    df['df_centre_backs'] = (d* df.Positioning +d * df.Reactions + c * df.Interceptions + d * df.Sliding_Tackle + d * df.Standing_Tackle + b * df.Vision + b * df.Composure + b * df.Crossing + a * df.Short_Pass + b * df.Long_Pass + c * df.Acceleration + b * df.Speed+ d * df.Stamina + d * df.Jumping + d * df.Heading + b * df.Long_Shots + d * df.Marking + c * df.Aggression) / (6 * b + 3 * c + 8 * d)
    df['df_wb_Wing_Backs'] = (b * df.Ball_Control + a * df.Dribbling + a * df.Marking + d * df.Sliding_Tackle + d * df.Standing_Tackle + c * df.Vision + c * df.Crossing + b * df.Short_Pass + c * df.Long_Pass + d * df.Acceleration + d * df.Speed + c * df.Stamina + a * df.Finishing) / (3 * a + 2 * b + 4 * c + 4 * d)

    lcb = df[(df['Club_Position'] == 'LCB')|(df['Club_Position'] == 'CB')].sort_values('df_centre_backs', ascending=False)[:4]
    x1Name = np.array(list(lcb['Name']))
    x1Photo = np.array(list(lcb['Photo']))
    x1Number = np.array(list(lcb['Club_Number']))

    rcb = df[(df['Club_Position'] == 'RCB')|(df['Club_Position'] == 'CB')].sort_values('df_centre_backs', ascending=False)[:4]
    x2Name = np.array(list(rcb['Name']))
    x2Photo = np.array(list(rcb['Photo']))
    x2Number = np.array(list(rcb['Club_Number']))

    lwb = df[(df['Club_Position'] == 'LWB') | (df['Club_Position'] == 'LB')].sort_values('df_wb_Wing_Backs', ascending=False)[:4]
    x3Name = np.array(list(lwb['Name']))
    x3Photo = np.array(list(lwb['Photo']))
    x3Number = np.array(list(lwb['Club_Number']))

    rwb = df[(df['Club_Position'] == 'RWB') | (df['Club_Position'] == 'RB')].sort_values('df_wb_Wing_Backs',ascending=False)[:4]
    x4Name = np.array(list(rwb['Name']))
    x4Photo = np.array(list(rwb['Photo']))
    x4Number = np.array(list(rwb['Club_Number']))

    return x1Name,x1Photo,x1Number,x2Name,x2Photo,x2Number,x3Name,x3Photo,x3Number,x4Name,x4Photo,x4Number

def find_BestMidFielders(df):
    # weights
    a = 0.5
    b = 1
    c = 2
    d = 3

    df['mf_playmaker'] = (d * df.Ball_Control + d * df.Dribbling + a * df.Marking + d * df.Reactions + d * df.Vision + c * df.Crossing + d * df.Short_Pass + c * df.Long_Pass + c * df.Curve + b * df.Long_Shots + c * df.Freekick_Accuracy) / (1 * a + 1 * b + 3 * c + 4 * d)
    df['mf_beast'] = ( d * df.Agility + c * df.Balance + b * df.Jumping + c * df.Strength + d * df.Stamina + a * df.Speed + c * df.Acceleration + d * df.Short_Pass + c * df.Aggression + d * df.Reactions + b * df.Marking + b * df.Standing_Tackle + b * df.Sliding_Tackle + b * df.Interceptions) / (1 * a + 5 * b + 4 * c + 4 * d)
    df['mf_controller'] = (b * df.Weak_foot + d * df.Ball_Control + a * df.Dribbling + a * df.Marking + a * df.Reactions + c * df.Vision + c * df.Composure + d * df.Short_Pass + d * df.Long_Pass) / (2 * c + 3 * d + 4 * a)

    plt.figure(figsize=(15, 6))

    cam = df[(df['Club_Position'] == 'CM')|(df['Club_Position'] == 'CAM') | (df['Club_Position'] == 'LAM') | (df['Club_Position'] == 'RAM')].sort_values('mf_playmaker', ascending=False)[:4]
    x1Name = np.array(list(cam['Name']))
    x1Photo = np.array(list(cam['Photo']))
    x1Number = np.array(list(cam['Club_Number']))

    rcm = df[(df['Club_Position'] == 'RDM')|(df['Club_Position'] == 'CDM')|(df['Club_Position'] == 'RCM')|(df['Club_Position'] == 'CM')].sort_values('mf_beast', ascending=False)[:4]
    x2Name = np.array(list(rcm['Name']))
    x2Photo = np.array(list(rcm['Photo']))
    x2Number = np.array(list(rcm['Club_Number']))

    lcm = df[(df['Club_Position'] == 'LDM')|(df['Club_Position'] == 'CDM')|(df['Club_Position'] == 'LCM')|(df['Club_Position'] == 'CM')].sort_values('mf_controller',ascending=False)[:4]
    x3Name = np.array(list(lcm['Name']))
    x3Photo = np.array(list(lcm['Photo']))
    x3Number = np.array(list(lcm['Club_Number']))
    return x1Name,x1Photo,x1Number,x2Name,x2Photo,x2Number,x3Name,x3Photo,x3Number

def find_BestAttackers(df):
    # weights
    a = 0.5
    b = 1
    c = 2
    d = 3

    df['att_left_wing'] = (c * df.Weak_foot + c * df.Ball_Control + c * df.Dribbling + c * df.Speed + d * df.Acceleration + b * df.Vision + c * df.Crossing + b * df.Short_Pass + b * df.Long_Pass + b * df.Aggression + b * df.Agility + a * df.Curve + c * df.Long_Shots + b * df.Freekick_Accuracy + d * df.Finishing) / (a + 6 * b + 6 * c + 2 * d)
    df['att_right_wing'] = ( c * df.Weak_foot + c * df.Ball_Control + c * df.Dribbling + c * df.Speed + d * df.Acceleration + b * df.Vision + c * df.Crossing + b * df.Short_Pass + b * df.Long_Pass + b * df.Aggression + b * df.Agility + a * df.Curve + c * df.Long_Shots + b * df.Freekick_Accuracy + d * df.Finishing) / ( a + 6 * b + 6 * c + 2 * d)
    df['att_striker'] = ( b * df.Weak_foot + b * df.Ball_Control + a * df.Vision + b * df.Aggression + b * df.Agility + a * df.Curve + a * df.Long_Shots + d * df.Balance + d * df.Finishing + d * df.Heading + c * df.Jumping + c * df.Dribbling) / (3 * a + 4 * b + 2 * c + 3 * d)

    lw = df[(df['Club_Position'] == 'LW') | (df['Club_Position'] == 'LM') | (df['Club_Position'] == 'LS')|(df['Club_Position'] == 'LF')].sort_values('att_left_wing', ascending=False)[:4]
    x1Name = np.array(list(lw['Name']))
    x1Photo = np.array(list(lw['Photo']))
    x1Number = np.array(list(lw['Club_Number']))

    rw = df[(df['Club_Position'] == 'RW') | (df['Club_Position'] == 'RM') | (df['Club_Position'] == 'RS')|(df['Club_Position'] == 'RF')].sort_values('att_right_wing', ascending=False)[:4]
    x2Name = np.array(list(rw['Name']))
    x2Photo = np.array(list(rw['Photo']))
    x2Number = np.array(list(rw['Club_Number']))

    plt.figure(figsize=(15, 6))
    st = df[(df['Club_Position'] == 'ST') | (df['Club_Position'] == 'LS') | (df['Club_Position'] == 'RS') | (df['Club_Position'] == 'CF')].sort_values('att_striker', ascending=False)[:4]
    x3Name = np.array(list(st['Name']))
    x3Photo = np.array(list(st['Photo']))
    x3Number = np.array(list(st['Club_Number']))

    return x1Name,x1Photo,x1Number,x2Name,x2Photo,x2Number,x3Name,x3Photo,x3Number

@app.route("/recommend", methods=['GET'])
def recommend():
    dataset = read_Dataset()
    teamName = request.args.get('team')
    dataset = get_ClubPlayers(dataset, teamName)
    gssName,gssPhoto,gssNumber,gsName,gsPhoto,gsNumber=find_BestGoalkeeper(dataset)
    lcbName,lcbPhoto,lcbNumber,rcbName,rcbPhoto,rcbNumber,lwbName,lwbPhoto,lwbNumber,rwbName,rwbPhoto,rwbNumber=find_BestDefenders(dataset)
    camName,camPhoto,camNumber,rcmName,rcmPhoto,rcmNumber,lcmName,lcmPhoto,lcmNumber=find_BestMidFielders(dataset)
    lwName,lwPhoto,lwNumber,rwName,rwPhoto,rwNumber,stName,stPhoto,stNumber=find_BestAttackers(dataset)
    return  jsonify({"Formation":[
        {"GoalKeeper":
            {
                "Name":gssName[0],
                "Photos":gssPhoto[0],
                "Numbers":gssNumber[0],
                "Position":"GOALKEEPER"
             },
        },
        {"Left Central Defender":
            {
                "Name": lcbName[0],
                "Photos": lcbPhoto[0],
                "Numbers": lcbNumber[0],
                "Position": "Left Central Defender"
            },
        },
        {"Right Central Defender":
            {
                "Name": rcbName[0],
                "Photos": rcbPhoto[0],
                "Numbers": rcbNumber[0],
                "Position": "Right Central Defender"
            },
        },
        {"Left Wing Back":
            {
                "Name": lwbName[0],
                "Photos": lwbPhoto[0],
                "Numbers": lwbNumber[0],
                "Position": "Left Wing Back"
            },
        },
        {"Right Wing Back":
            {
                "Name": rwbName[0],
                "Photos": rwbPhoto[0],
                "Numbers": rwbNumber[0],
                "Position": "Right Wing Back"
            },
        },
        {"PlayMaker Mid_Fielders":
            {
                "Name": camName[0],
                "Photos": camPhoto[0],
                "Numbers": camNumber[0],
                "Position": "PlayMaker Mid_Fielders"
            },
        },
        {"Beast Mid_Fielders":
            {
                "Name": rcmName[0],
                "Photos": rcmPhoto[0],
                "Numbers": rcmNumber[0],
                "Position": "Beast Mid_Fielders"
            },
        },
        {"Controller Mid_Fielders":
            {
                "Name": lcmName[0],
                "Photos": lcmPhoto[0],
                "Numbers": lcmNumber[0],
                "Position": "Controller Mid_Fielders"
            },
        },
        {"Left Wing Attacker":
            {
                "Name": lwName[0],
                "Photos": lwPhoto[0],
                "Numbers": lwNumber[0],
                "Position": "Left Wing Attacker"
            },
        },
        {"Right Wing Attacker":
            {
                "Name": rwName[0],
                "Photos": rwPhoto[0],
                "Numbers": rwNumber[0],
                "Position": "Right Wing Attacker"
            },
        },
        {"Striker Attacker":
            {
                "Name": stName[0],
                "Photos": stPhoto[0],
                "Numbers": stNumber[0],
                "Position": "Striker Attacker"
            },
        },
        ]
    })



if __name__ == '__main__':
    app.run(debug=True)
