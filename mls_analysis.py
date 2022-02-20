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

def preprocess_players(dataset: pd.DataFrame) -> pd.DataFrame:
    #Quitar equipos que no aparezcan en los salarios
    #Estandarizar nombres (quitar espacios y cosas así)
    #Quitar columnas que no usaremos
    return dataset

if __name__ == "__main__":
    # salarie_data = getYearSalary(2017)
    # salarie_data = preprocess_salaries(salarie_data)
    # salarie_data.dropna(inplace=True)
    diff_teams = set()

    players = load_data(PLAYER_DATASET)
    players = limit_to_year_range(players,2007,2017)
    players.dropna(inplace=True)
    for i in players["Club"]:
        i = str(i).strip()
    print(players['Club'].unique())
    

