import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

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
TEAM_DATASET = "./all_tables.csv"
SALARIES_DATASET = "./mls-salaries-"  # iterate to get the year
COLUMNS_PLAYERS = ["Player", "Club", "POS", "Year", "GP", "MINS"]

FIGURES_DIRECTORY = './figures'

POSITION_DICT = {
    "F": ["F", "M-F", "F-M"],
    "M": ["M", "M-F", "M-D"],
    "D": ["D", "M-D", "D-M"],
}

# If you want to add-delete more stats to check, modify the next dictionary
STATS_DICT = {
    "F": ["G", "A", "SOG", "GWG", "G/90min"],
    "M": ["A", "GWA", "A/90min", "G"],
    "D": ["A", "FC", "A/90min", "G"]
}

DEFAULT_STATS = ["G", "A", "SOG", "FC", "G/90min", "A/90min"]

MLS_TEAMS_DICT = {
    "ATL": "Atlanta United Football Club",
    "CHI": "Chicago Fire",
    "CLB": "Columbus Crew SC",
    "COL": "Colorado Rapids",
    "DAL": "FC Dallas",
    "DC": "D.C. United",
    "HOU": "Houston Dynamo",
    "KC": "Sporting Kansas City",
    "TOR": "Toronto FC",
    "RSL": "Real Salt Lake",
    "NYRB": "New York Red Bulls",
    "MNUFC": "Minnesota United FC",
    "SEA": "Seattle Sounders FC",
    "PHI": "Philadelphia Union",
    "NE": "New England Revolution",
    "POR": "Portland Timbers",
    "VAN": "Vancouver Whitecaps",
    "NYCFC": "New York City FC",
    "LA": "LA Galaxy",
    "NY": "New York City FC",
    "TFC": "Toronto FC",
    "MTL": "CF Montreal",
    "LAFC": "Los Angeles Football Club",
    "ORL": "Orlando City Soccer Club",
    "CHV": "Chivas USA",
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
    "F": "Forward",  # Striker, Center Forward, Wingers
    "M": "Midfielder",  # Left Mid, Right Mid, Center Mid, Attacking Mid, Defensive Mid
    "M-F": "Midfielder-Forward",
    # Center back, Full Back (Left Back, Right Back), Wing Backs, Sweeper (RARE)
    "D": "Defender",
    "M-D": "Midfielder-Defender",
    "D-M": "Defender-Midfielder",
    "F-M": "Forward-Midfielder",
    "A": "All positions"
}

STATS_MEANING_DICT = {
    "G":    "Goals",
    "A":    "Assists",
    "SOG%": "Percentage of Goals per Shot on Target",
    "SOG":  "Shot on Target",
    "GWG":  "Game Winning Goal",
    "GWA":  "Game Winning Assist",
    "A/90min":  "Assists/ 90 min played",
    "G/90min":  "Goals/ 90 min played",
    "FC": "Fouls Commited"
}

# Loads the .csv mls salaries file corresponding to the year provided


def get_year_salary(year: int) -> pd.DataFrame:
    if year < 2007 or year > 2017:
        return None
    return pd.read_csv(str(SALARIES_DATASET+str(year)+".csv"))

# Loads the csv file from the path provided


def load_data(csv_file_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_file_path)
    return data

# limits the dataset to the interval of years provided


def limit_to_year_range(dataset: pd.DataFrame, start_year: int, stop_year: int) -> pd.DataFrame:
    dataset = dataset[dataset.Year > start_year - 1]
    dataset = dataset[dataset.Year < stop_year + 1]
    return dataset

# Combines the two names fields into a single field.
# Homologues the "club" field name with the one from player stats
# Optional param to limit the salary to a specific team.


def preprocess_salaries(data: pd.DataFrame, team: str = None) -> pd.DataFrame:
    dataset = data.copy()
    dataset['Player'] = dataset['first_name'] + ' ' + dataset['last_name']
    dataset.drop(columns=["first_name", "last_name", "position"], inplace=True)
    dataset.rename(columns={'club': "Club"}, inplace=True)
    if team is not None:
        for x in dataset.index:
            if dataset.loc[x, "Club"] != team:
                dataset.drop(x, inplace=True)
    return dataset

# Get the list of teams from the .csv salaries files.


def get_teams_from_salaries() -> list:
    teams = []
    for year in range(2007, 2018):
        data = get_year_salary(year)
        data.dropna(inplace=True)
        for team in data['club'].unique():
            if str(team) not in teams:
                teams.append(str(team))
    return teams

# pseudo-switch-case to change the name of teams that are the same with different abbreviations


def normalize_team(team: str) -> str:
    switcher = {
        "MIN": "MNUFC",
        "NYC": "NYCFC",
        "NY": "NYCFC",
        "NYR": "NYRB",
        "RBNY": "NYRB",
        "SKC": "KC",
        "TOR": "TFC"
    }

    return switcher.get(team, team)

# limits player data to the years from which we have info
# remove teams that doesn't appear in the salaries files
# normalize team names in the file


def preprocess_players(dataset: pd.DataFrame, team: str = None) -> pd.DataFrame:
    # Quitar equipos que no aparezcan en los salarios
    dataset = limit_to_year_range(dataset, 2007, 2017)
    teams = get_teams_from_salaries()
    for x in dataset.index:
        # Remove spaces and normalize teams with different abreviation
        dataset.loc[x, "Club"] = normalize_team(
            str(dataset.loc[x, "Club"]).strip())
        if dataset.loc[x, "Club"] not in teams:
            dataset.drop(x, inplace=True)
            continue
        if team is not None and team in teams:
            if dataset.loc[x, "Club"] != team:
                dataset.drop(x, inplace=True)

    return dataset
# Quitar columnas que no usaremos
# Which columns are more important?
#Player, Club, Position,
# For each position, different stats.
# Forward, Mid-Forward, Forward-Mid -> Goals, Assist, SOG%
# Midfielder, Mid-Forward, Mid-Defender -> Assist, GWA, A/90min, Goals(?)
# Defender, Mid-Defender, Defender-Mid -> Assists, Fouls Commited (negative relation?), A/90min, Goals(?)

# limit a dataframe by one specific position


def limit_by_position(dataset: pd.DataFrame, position: str) -> pd.DataFrame:

    data = dataset.copy()

    if position not in POSITION_DICT.keys():
        print("Position not in list")
        return data

    cols = [x for x in list(data.columns) if x not in STATS_DICT.get(
        position, DEFAULT_STATS) + COLUMNS_PLAYERS]

    data.drop(cols, axis=1, inplace=True)

    for x in data.index:
        if data.loc[x, "POS"] not in POSITION_DICT.get(position):
            data.drop(x, inplace=True)

    return data

# Matches the player data stats with the salary info on them according to year.
# Can match only a specific team if desired


def combine_players_salaries(players_data: pd.DataFrame, year: int, team: str = None) -> pd.DataFrame:
    players_data = limit_to_year_range(players_data, year, year)
    salary_data = get_year_salary(year)
    salary_data = preprocess_salaries(salary_data, team)
    data = pd.merge(players_data, salary_data, on=["Player", "Club"])

    return data

# Generates visualization according to a specific param vs. base_salary std deviation of each team.
# Plot made by year and position.
# The plot is saved in the computer.


def visualization(dataset: pd.DataFrame, param: str, year: int, position: str):
    dataset = combine_players_salaries(dataset, year)
    # Standard Deviation of Base Salary per team.
    salaries_per_team = dataset[["Club", "base_salary"]].groupby([
                                                                 "Club"]).std()
    # Sum of the param given in the argument per team.
    goals_per_team = dataset[["Club", param]].groupby(["Club"]).sum()

    # We merge both params per Club to get the DataFrame for the plot.
    mix = pd.merge(salaries_per_team, goals_per_team, on="Club")

    if mix.empty:
        return

    # We start the plot.
    fig, ax = plt.subplots()
    for team in mix.index:
        ax.plot(mix.loc[team, "base_salary"], mix.loc[team,
                param], label=team, marker='o', linestyle='')
    ax.legend(ncol=2)

    # Create the directory (If it doesn't exist)
    try:
        if not os.path.exists(f'{FIGURES_DIRECTORY}'):
            os.mkdir(FIGURES_DIRECTORY)
        if not os.path.exists(f'{FIGURES_DIRECTORY}/{year}'):
            os.mkdir(f'{FIGURES_DIRECTORY}/{year}')
        if not os.path.exists(f'{FIGURES_DIRECTORY}/{year}/{POS_DICT.get(position)}'):
            os.mkdir(f'{FIGURES_DIRECTORY}/{year}/{POS_DICT.get(position)}')
            print(
                "Directory "+f'{FIGURES_DIRECTORY}/{year}/{POS_DICT.get(position)}'+" created")
    except OSError as error:
        print(error)

    title = STATS_MEANING_DICT.get(
        param, param) + " vs. Salary Standard Deviation for " + POS_DICT.get(position) + " " + str(year)

    plt.title(title)
    plt.xlabel("Base Salary Standard Deviation")
    plt.ylabel(STATS_MEANING_DICT.get(param) + " Sum")
    temp = param
    if '/' in temp:
        temp = temp.replace('/', '_per_')
    plt.savefig(
        f'{FIGURES_DIRECTORY}/{year}/{POS_DICT.get(position)}/{temp}_{position}.png')
    plt.close()

# options:
#  0 => Champions and winners of Supporters' Shield
#  1 => Champions Only
#  2 => Supporters' Shield's Winners
def get_champions_data(year: int = None,option: int = 0) -> pd.DataFrame:
    data = load_data(TEAM_DATASET)
    data = limit_to_year_range(data, 2007, 2017)

    # abv_dict = {
    #     "(X)":  "Supporters Shield",
    #     "(SS)": "Supporters Shield",
    #     "(C)":  "Champion",
    #     "(W1)": "Western Confederation Champion",
    #     "(V)":  "Canadian Champion Winner",
    #     "(U)":  "US Open Cup Winner",
    #     "(E1)": "Eastern Confederation Champion",
    #     "(C1)": "Central Confederation Champion"
    # }

    relevant_pattern = r'\([CUEXSCWV]{1,2}1?(\, )?([SSXEWC]1?)?\)'
    champion_pattern = r'\(([CWEUV]1?(\, [XS]{1,2})?|[XS]{1,2}\, [CWE]1?)\)'
    shield_pattern = r'\(([XS]{1,2}(\, [A-Z]1?)?|[CEW]1?\, [XS]{1,2})\)'
    options = [relevant_pattern,champion_pattern,shield_pattern]

    data = data[data["Conference"] == "Overall"]
    if year is not None:
        data = data[data["Year"] == year]
    # data = data[re.search(champion_pattern,str(data["Team"])) is not None]
    for x in data.index:
        if re.search(options[option], str(data.loc[x, "Team"])) is None:
            data.drop(x, inplace=True)

    return data


def get_team_abv(team: str) -> str:

    reverse_dict = {'D.C. United': 'DC',
                    'Chivas USA' : 'CHV',
                    'Houston Dynamo': 'HOU',
                    'New England Revolution': 'NE',
                    'FC Dallas': 'DAL',
                    'New York Red Bulls': 'NYRB',
                    'Chicago Fire': 'CHI',
                    'Kansas City Wizards': 'KC',
                    'Columbus Crew': 'CLB',
                    'Colorado Rapids': 'COL',
                    'LA Galaxy': 'LA',
                    'Real Salt Lake': 'RSL',
                    'Toronto FC 1': 'TOR',
                    'Toronto FC': 'TFC',
                    'San Jose Earthquakes': 'SJ',
                    'Seattle Sounders FC': 'SEA',
                    'Philadelphia Union': 'PHI',
                    'Sporting Kansas City': 'KC',
                    'Portland Timbers': 'POR',
                    'Vancouver Whitecaps FC': 'VAN',
                    'Montreal Impact': 'MTL',
                    'Orlando City SC': 'ORL',
                    'New York City FC': 'NYCFC',
                    'Columbus Crew SC': 'CLB',
                    'Atlanta United FC': 'ATL',
                    'Minnesota United FC': 'MNUFC'}
    res = ""
    if team not in reverse_dict.keys():
        words = team.split(" ")
        for i in words:
            res += i[0]
    return reverse_dict.get(team,res)

def normalize_team_names(table_data: pd.DataFrame) -> pd.DataFrame:
    copy = table_data.copy()
    for x in copy.index:
        copy.loc[x,"Team"] = get_team_abv(str(copy.loc[x,"Team"]).split(" (")[0])
    copy.rename(columns={"Team":"Club"},inplace=True)
    return copy

def preprocess_table(data: pd.DataFrame, year: int = None, club: str = None) -> pd.DataFrame:
    champion_pattern = r'\(([CWEUV]1?(\, [XS]{1,2})?|[XS]{1,2}\, [CWE]1?)\)'
    shield_pattern = r'\(([XS]{1,2}(\, [A-Z]1?)?|[CEW]1?\, [XS]{1,2})\)'

    dataset = data.copy()

    if year is not None:
        dataset = limit_to_year_range(dataset,year,year)
    else:
        dataset = limit_to_year_range(dataset,2007,2017)

    dataset = dataset[dataset["Conference"] == "Overall"]
    
    for x in dataset.index:
        dataset.loc[x,"Champion"] = re.search(champion_pattern, str(dataset.loc[x, "Team"])) is not None
        dataset.loc[x,"Shield"]   = re.search(shield_pattern  , str(dataset.loc[x, "Team"])) is not None
    
    # dataset["Champion"] = re.search(champion_pattern, str(dataset.Team)) is not None
    # dataset["Shield"]   = re.search(shield_pattern, str(dataset.Team)) is not None
    dataset = normalize_team_names(dataset)
    
    dataset.drop(columns=["SW","SL","D","Qualification","Conference","Head-to-head","PPG"],inplace=True)
    
    if club is not None:
        dataset = dataset[dataset.Club == club]
    
    return dataset


# Things to consider:
# -Get statistics of the full team
# -Add base_salary + compensation and get std deviation
# -Check with team performance:
#   - Goals? Wins? Assists? SOG?


if __name__ == "__main__":

    # #We start analyzing the data,
    #  generating charts
    # for year in range(2007,2018):
    #     for position in STATS_DICT.keys():
    #         for param in STATS_DICT.get(position):
    #             data_per_position = limit_by_position(players, position)
    #             visualization(data_per_position,param,year,position)

    table = load_data(TEAM_DATASET)
    table = preprocess_table(table)
    print(table.head())
