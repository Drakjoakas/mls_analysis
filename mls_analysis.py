from json import load
from tokenize import String
from typing import Set
import pandas as pd

#  =====Datasets====
#   "all_players.csv": all outfield player data from every season of MLS through 2020 (1996-2020)
#   Columns:
#   -Player
#   -Club
#   -POS position
#   -GP games played
#   -GS games started
#   -MINS minutes played
#   -G goals
#   -A assist
#   -SHTS shots
#   -SOG shots on target
#   -GWG game winning goal (goal that leave a team one goal ahead)
#   -PKG/A penalty kick goals / attempts
#   -HmG Home Goals
#   -RdG Road Goals
#   -G/90min Goals for every 90 min played
#   -SC%
#   -GWA game winning assist (assist that leave a team one goal ahead)
#   -HmA Home Assists
#   -RdA Road Assists
#   -A/90min Assists for every 90 min
#   -FC fouls commited  (free kick for opposit team)
#   -FS fouls sustained (free kick for his team)
#   -OFF offside
#   -YC (?)
#   -RC (?)
#   -SOG% Goals scored per shot on goal
#   -Year
#   -Season
#
#   "mls-salaries-20XX.csv":
#   Columns:
#   -Club
#   -last_name
#   -first_name
#   -position
#   -base_salary
#   -guaranteed_compensation
#   

PLAYER_DATASET = "./all_players.csv"
SALARIES_DATASET = "./mls-salaries-"    #iterate to get the year

MLS_TEAMS_DICT = {
    "ATL" : "Atlanta United Football Club",
    "CHI" : "Chicago Fire",
    "CLB" : "Columbus Crew SC",
    "COL" : "Colorado Rapids",
    "DAL" : "FC Dallas",
    "DC"  : "D.C. United",
    "HOU" : "Houston Dynamo",
    "KC"  : "Sporting Kansas City",
    "TOR" : "Toronto FC",
    "RSL" : "Real Salt Lake",
    "NYRB": "New York Red Bulls",
    "MNUFC": "Minnesota United FC",
    "SEA" : "Seattle Sounders FC",
    "PHI" : "Philadelphia Union",
    "NE"  : "New England Revolution",
    "POR" : "Portland Timbers",
    "VAN" : "Vancouver Whitecaps",
    "NYCFC": "New York City FC",
    "LA": "LA Galaxy",
    "NY": "New York City FC",
    "TFC": "Toronto FC",
    "MTL": "CF Montreal",
    "LAFC": "Los Angeles Football Club",
    "ORL": "Orlando City Soccer Club",
    "CHV" : "Chivas USA",
    # "SJ",
    # "MIA",
    # "NSH",
    # "PAN",
    # "SKC",
    # "SLV",
    # "MIN":"Minnessota?",      Estandarizar
    # "HAI",
    # "USA",
    # "ROC",
    # "ECU",
    # "HON",
    # "CIN",
    # "NYC": "New York City RB",    Estandarizar
    # "NYR": "New York City Red Bull", Estandarizar
    # "CAN",
    # "RBNY": "New York Red Bull", Estandarizar
    # "JAM"
}

POS_DICT = {
    "F": "Forward",    #Striker, Center Forward, Wingers
    "M": "Midfielder", #Left Mid, Right Mid, Center Mid, Attacking Mid, Defensive Mid
    "M-F": "Midfielder-Forward",
    "D": "Defender",  #Center back, Full Back (Left Back, Right Back), Wing Backs, Sweeper (RARE)
    "M-D": "Midfielder-Defender",
    "D-M": "Defender-Midfielder",
    "F-M": "Forward-Midfielder"
}



def getYearSalary(year: int) -> pd.DataFrame:
    if year < 2007 or year > 2017:
        return None
    return pd.read_csv(str(SALARIES_DATASET+str(year)+".csv"))

def load_data(csv_file_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_file_path)
    return data

def limit_to_year_range(dataset: pd.DataFrame, start_year: int,stop_year: int) -> pd.DataFrame:
    dataset = dataset[dataset.Year > start_year - 1]
    dataset = dataset[dataset.Year < stop_year  + 1]
    return dataset

def preprocess_salaries(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset['name'] = dataset['first_name'] + ' ' + dataset['last_name']
    return dataset

def get_teams_from_salaries() -> list:
    teams = []
    for year in range (2007,2018):
        data = getYearSalary(year)
        data.dropna(inplace=True)
        for team in data['club'].unique():
            if str(team) not in teams:
                teams.append(str(team))
    return teams 

def normalize_team(team: str) -> str:
    switcher = {
        "MIN" : "MNUFC",
        "NYC" : "NYCFC",
        "NY" : "NYCFC",
        "NYR" : "NYRB",
        "RBNY": "NYRB"
    }
    
    return switcher.get(team,team)

def preprocess_players(dataset: pd.DataFrame) -> pd.DataFrame:
    #Quitar equipos que no aparezcan en los salarios
    teams = get_teams_from_salaries()
    for x in dataset.index:
        dataset.loc[x,"Club"] = normalize_team(str(dataset.loc[x,"Club"]).strip())  #Remove spaces and normalize teams with different abreviation
        if dataset.loc[x,"Club"] not in teams:
            dataset.drop(x,inplace=True)
            
    #Quitar columnas que no usaremos
    #Which columns are more important?
    #Player, Club, Position,
    #For each position, different stats.
    # Forward, Mid-Forward, Forward-Mid -> Goals, Assist, SOG%
    # Midfielder, Mid-Forward, Mid-Defender -> Assist, GWA, A/90min, Goals(?)
    # Defender, Mid-Defender, Defender-Mid -> Assists, Fouls Commited (negative relation?), A/90min, Goals(?)
    return dataset

if __name__ == "__main__":
    # salarie_data = getYearSalary(2017)
    # salarie_data = preprocess_salaries(salarie_data)
    # salarie_data.dropna(inplace=True)
    
    # diff_teams = set()

    players = load_data(PLAYER_DATASET)
    players = limit_to_year_range(players,2007,2017)
    players.dropna(inplace=True)
    print(players["POS"].unique())

    
    

