import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as mark
import numpy as np
import os
import re
import math
import random
from scipy import stats as st

#  =====Datasets====
#   "all_players.csv": all outfield player data from every season of MLS through 2020 (1996-2020)
#   @url: https://www.kaggle.com/josephvm/major-league-soccer-dataset 
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
#   "all_tables.csv":
#   Columns:
#   - POS table position
#   - Team
#   - GP Games Played
#   - W Wins
#   - L Loss
#   - SW
#   - GF Goals for
#   - GA Goals Against
#   - GD Goals Difference
#   - Pts Points
#   - Qualification
#   - Conference
#   - Year
#   - SL
#   - D
#   - Head-to-head
#   - PPG
#
#   "mls-salaries-20XX.csv":
#   @url: https://www.kaggle.com/crawford/us-major-league-soccer-salaries
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

MIN_MINS_PLAYED = 500
MIN_GAMES_PLAYED = 5

AMS_DIM = 3
K = 7
K_MEAN_STEPS = 30
ANALYSIS_BAR = 15

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

DEFAULT_STATS = ["G", "A", "SOG", "G/90min", "A/90min", "GWG", "GWA"]

MLS_TEAMS_DICT = {
    "ATL"   : "Atlanta United Football Club",
    "CHI"   : "Chicago Fire",
    "CLB"   : "Columbus Crew SC",
    "COL"   : "Colorado Rapids",
    "DAL"   : "FC Dallas",
    "DC"    : "D.C. United",
    "HOU"   : "Houston Dynamo",
    "KC"    : "Sporting Kansas City",
    "TOR"   : "Toronto FC",
    "RSL"   : "Real Salt Lake",
    "NYRB"  : "New York Red Bulls",
    "MNUFC" : "Minnesota United FC",
    "SEA"   : "Seattle Sounders FC",
    "PHI"   : "Philadelphia Union",
    "NE"    : "New England Revolution",
    "POR"   : "Portland Timbers",
    "VAN"   : "Vancouver Whitecaps",
    "NYCFC" : "New York City FC",
    "LA"    : "LA Galaxy",
    "NY"    : "New York City FC",
    "TFC"   : "Toronto FC",
    "MTL"   : "CF Montreal",
    "LAFC"  : "Los Angeles Football Club",
    "ORL"   : "Orlando City Soccer Club",
    "CHV"   : "Chivas USA"
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


REG_SEASON_WEST = {
    2007: "CHV",
    2008: "HOU",
    2009: "LA",
    2010: "LA",
    2011: "LA",
    2012: "SJ",
    2013: "POR",
    2014: "SEA",
    2015: "POR",
    2016: "SEA",
    2017: "TFC"
    }
REG_SEASON_EAST = {
    2007: "DC",
    2008: "CLB",
    2009: "CLB",
    2010: "NYRB",
    2011: "KC",
    2012: "KC",
    2013: "NYRB",
    2014: "DC",
    2015: "POR",
    2016: "SEA",
    2017: "TFC"
    }

EVALUATION_WINS_2018 = {
  "NYRB": 22,
  "ATL": 21,
  "KC": 18,
  "SEA": 18,
  "LAF": 16,
  "DAL": 16,
  "NYCFC": 16,
  "POR": 15,
  "DC": 14,
  "CLB": 14,
  "PHI": 15,
  "RSL": 14,
  "LA": 13,
  "VAN": 13,
  "MTL": 14,
  "NE": 10,
  "HOU": 10,
  "MNUFC": 11,
  "TFC": 10,
  "CHI": 8,
  "COL": 8,
  "ORL": 8,
  "SJ": 4
}

EVALUATION_WINS_2019 = {
  "LAF": 21,
  "NYCFC": 18,
  "ATL": 18,
  "SEA": 16,
  "PHI": 16,
  "RSL": 16,
  "MNUFC": 15,
  "LA": 16,
  "TFC": 13,
  "DC": 13,
  "POR": 14,
  "NYRB": 14,
  "DAL": 13,
  "NE": 11,
  "SJ": 13,
  "COL": 12,
  "CHI": 10,
  "MTL": 12,
  "HOU": 12,
  "CLB": 10,
  "KC": 10,
  "ORL": 9,
  "VAN": 8,
  "FC": 6
}

EVALUATION_WINS_2020 = {
  "PHI": 14,
  "TFC": 13,
  "KC": 12,
  "CLB": 12,
  "ORL": 11,
  "SEA": 11,
  "NYCFC": 12,
  "POR": 11,
  "MNUFC": 9,
  "COL": 8,
  "DAL": 9,
  "LAF": 9,
  "NYRB": 9,
  "NS": 8,
  "NE": 8,
  "SJ": 8,
  "VAN": 9,
  "MTL": 8,
  "IMC": 7,
  "LA": 6,
  "RSL": 5,
  "CFF": 5,
  "ATL": 6,
  "DC": 5,
  "HOU": 4,
  "FC": 4
}

class Centroid:
    def __init__(self, dim_list):
        self.place = dim_list

AVG = 'avg'
MED = 'median'
RAT = 'balance-score'
NORM_RAT = 'normalized_ratio'
DIST = 'distance'
VAR = 'variance'
GINI = 'Gini'


def get_year_salary(year: int) -> pd.DataFrame:
    """ Loads the .csv mls salaries file corresponding to the year provided."""

    if year < 2007 or year > 2017:
        return None
    return pd.read_csv(str(SALARIES_DATASET+str(year)+".csv"))

def get_full_salaries() -> pd.DataFrame:
    """Returns DataFrame with the data of MLS salaries of all the years.

    Returns:
        pd.DataFrame: DataFrame with salary per player, per year.
    """
    res = pd.DataFrame()
    for year in range(2007, 2018):
        data = get_year_salary(year)
        data["Year"] = year
        res = pd.concat([res, data], ignore_index=True)
    res = preprocess_salaries(res)
    res.dropna(axis=0, inplace=True)
    return res

def load_data(csv_file_path: str) -> pd.DataFrame:
    """ Loads the csv file from the path provided."""
    data = pd.read_csv(csv_file_path)
    return data

def limit_to_year_range(dataset: pd.DataFrame, start_year: int, stop_year: int) -> pd.DataFrame:
    """Limits the dataset to the interval of years provided

    Args:
        dataset (pd.DataFrame): DataFrame containing data
        start_year (int): Start year of limit
        stop_year (int): End year of limit

    Returns:
        pd.DataFrame: Dataframe limited by the years provided
    """
    dataset = dataset[dataset.Year > start_year - 1]
    dataset = dataset[dataset.Year < stop_year + 1]
    return dataset

def preprocess_salaries(data: pd.DataFrame, team: str = None) -> pd.DataFrame:
    """Adds columns first_name and last_name to create field Player. 
       Homologues the "club" field name.

    Args:
        data (pd.DataFrame): Dataframe containg salarys data
        team (str, optional): Abv of the team if wanted to limit. Defaults to None.

    Returns:
        pd.DataFrame: Modified DataFrame 
    """
    dataset = data.copy()
    dataset['Player'] = dataset['first_name'] + ' ' + dataset['last_name']
    dataset['Wage'] = dataset['base_salary'] + \
        dataset['guaranteed_compensation']
    dataset.drop(columns=["first_name", "last_name", "position"], inplace=True)
    dataset.rename(columns={'club': "Club"}, inplace=True)
    if team is not None:
        for x in dataset.index:
            if dataset.loc[x, "Club"] != team:
                dataset.drop(x, inplace=True)
    return dataset

def get_teams_from_salaries() -> list:
    """Get the list of teams from the .csv salaries files.

    Returns:
        list: List of teams from the salary files
    """
    teams = []
    for year in range(2007, 2018):
        data = get_year_salary(year)
        data.dropna(inplace=True)
        for team in data['club'].unique():
            if str(team) not in teams:
                teams.append(str(team))
    return teams

def normalize_team(team: str) -> str:
    """pseudo-switch-case to change the name of teams that are the same with different abbreviations

    Args:
        team (str): Abv. of the team to transform

    Returns:
        str: Abv normalized
    """
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

# 


def preprocess_players(dataset: pd.DataFrame, team: str = None) -> pd.DataFrame:
    """limits player data to the years from which we have info
       remove teams that doesn't appear in the salaries files
       normalize team names in the file

    Args:
        dataset (pd.DataFrame): DataFrame from the players
        team (str, optional): Team name to limit the data. Defaults to None.

    Returns:
        pd.DataFrame: Modified dataFrame
    """
    dataset = limit_to_year_range(dataset, 2007, 2017)
    teams = get_teams_from_salaries()
    for x in dataset.index:
        dataset.loc[x, "Club"] = normalize_team(
            str(dataset.loc[x, "Club"]).strip())
        if dataset.loc[x, "Club"] not in teams:
            dataset.drop(x, inplace=True)
            continue
        if team is not None and team in teams:
            if dataset.loc[x, "Club"] != team:
                dataset.drop(x, inplace=True)

    dataset = dataset[dataset.MINS >= MIN_MINS_PLAYED]
    dataset = dataset[dataset.GP >= MIN_GAMES_PLAYED]

    return dataset




def limit_by_position(dataset: pd.DataFrame, position: str) -> pd.DataFrame:
    """# limit a dataframe by one specific position

    Args:
        dataset (pd.DataFrame): dataFrame containing players dataset
        position (str): Position to limit the dataFrame

    Returns:
        pd.DataFrame: Modified dataFrame
    """

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

def combine_players_salaries(players_data: pd.DataFrame, year: int, team: str = None) -> pd.DataFrame:
    """Matches the player data stats with the salary info on them according to year.
       Can match only a specific team if desired

    Args:
        players_data (pd.DataFrame): DataFrame of the players data
        year (int): Desired year to filter
        team (str, optional): Team bv to filter. Defaults to None.

    Returns:
        pd.DataFrame: Modified DataFrame
    """
    players_data = limit_to_year_range(players_data, year, year)
    salary_data = get_year_salary(year)
    salary_data = preprocess_salaries(salary_data, team)
    data = pd.merge(players_data, salary_data, on=["Player", "Club"])

    return data

# 


def visualization(dataset: pd.DataFrame, param: str, year: int, position: str):
    """Generates visualization according to a specific param vs. base_salary std deviation of each team.
# Plot made by year and position.
# The plot is saved in the computer.

    Args:
        dataset (pd.DataFrame): DataFrame containing necessary data
        param (str): param to plot
        year (int): year to be plot
        position (str): position of the player to be plot
    """
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


def get_team_abv(team: str) -> str:
    """Gets the team abreviation depending to the complete name

    Args:
        team (str): Full name of the team

    Returns:
        str: ABV of the team from dictionary or generated.
    """

    reverse_dict = {'D.C. United': 'DC',
                    'Chivas USA': 'CHV',
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
    return reverse_dict.get(team, res)


def normalize_team_names(table_data: pd.DataFrame) -> pd.DataFrame:
    """Standarize the team names from the Table Dataframe

    Args:
        table_data (pd.DataFrame): Dataframe containing the table data.

    Returns:
        pd.DataFrame: Modified table Data
    """
    copy = table_data.copy()
    for x in copy.index:
        copy.loc[x, "Team"] = get_team_abv(
            str(copy.loc[x, "Team"]).split(" (")[0])
    copy.rename(columns={"Team": "Club"}, inplace=True)
    return copy


def get_champions(data: pd.DataFrame) -> dict:
    """Obtain the champions from every Year

    Args:
        data (pd.DataFrame): DataFrame of the table position without normalized teams

    Returns:
        dict: Dictionary containing the champions of every
    """
    champion_pattern = r'\(([CWEUV]1?(\, [XS]{1,2})?|[XS]{1,2}\, [CWE]1?)\)'
    yearly_champions = dict()
    for year in range(2007, 2018):
        yearly_champions[year] = dict()
        for x in data[data.Year == year].index:
            if re.search(champion_pattern, str(data.loc[x, "Team"])) is not None:
                team = str(data.loc[x, "Team"]).split(" (")
                yearly_champions[year][team[1]] = get_team_abv(team[0])
    return yearly_champions


def preprocess_table(data: pd.DataFrame, year: int = None, club: str = None) -> pd.DataFrame:
    """Process the table data to standarize to the values worked

    Args:
        data (pd.DataFrame): Raw Data of the Table information
        year (int, optional): Year to limit data. Defaults to None.
        club (str, optional): Club name to limit data. Defaults to None.

    Returns:
        pd.DataFrame: Modified DataFrame. Limited by year and TeamNames normalized.
    """
    champion_pattern = r'\(([CWEUV]1?(\, [XS]{1,2})?|[XS]{1,2}\, [CWE]1?)\)'
    shield_pattern = r'\(([XS]{1,2}(\, [A-Z]1?)?|[CEW]1?\, [XS]{1,2})\)'

    dataset = data.copy()

    if year is not None:
        dataset = limit_to_year_range(dataset, year, year)
    else:
        dataset = limit_to_year_range(dataset, 2007, 2017)

    dataset = dataset[dataset["Conference"] == "Overall"]

    for x in dataset.index:
        dataset.loc[x, "Champion"] = re.search(
            champion_pattern, str(dataset.loc[x, "Team"])) is not None
        dataset.loc[x, "Shield"] = re.search(
            shield_pattern, str(dataset.loc[x, "Team"])) is not None
        if dataset.loc[x, "Champion"]:
            dataset.loc[x, "Num_Championships"] = 1
        else:
            dataset.loc[x, "Num_Championships"] = 0
    dataset = normalize_team_names(dataset)

    dataset.drop(columns=["SW", "SL", "D", "Qualification",
                 "Conference", "Head-to-head", "PPG"], inplace=True)

    if club is not None:
        dataset = dataset[dataset.Club == club]

    return dataset


def plot_salary_distribution(salary_dataset: pd.DataFrame, param="base_salary"):
    """Plots the income distribution using the data.

    Args:
        salary_dataset (pd.DataFrame): DataFrame containing the salary data
        param (str, optional): Param to plot the distribution. Defaults to "base_salary".
    """
    salary_dataset.sort_values(by=[param], inplace=True, ignore_index=True)
    salary_group_index = get_salary_groups(salary_dataset[param], 1.5)

    salary_axis = list()
    amount_axis = list()

    low = 0
    for index in salary_group_index:
        high = index
        norm = high - low
        amount_axis.append(norm)
        salary_axis.append(sum(salary_dataset.loc[low:high, param]) / norm)
        low = high
    plt.title("MLS Salary Distribution 2007 - 2017")
    plt.xlabel(param)
    plt.ylabel("Amount of players")
    plt.plot(salary_axis, amount_axis)
    plt.show()



def plot_salary_param_distribution(salary_dataset: pd.DataFrame, param_list: list) -> dict:
    """Plots salary distribution against a parameter list.

    Args:
        salary_dataset (pd.DataFrame): Dataframe containing the salary and parameters information
        param_list (list): List of parameters to plot against

    Returns:
        dict: Dictionary containing the correlation of the param
    """
    sorted_data = salary_dataset.sort_values(by=["Wage"], ignore_index=True)
    correlation_dict_pd = dict()
    correlation_dict_made = dict()
    x_axis = sorted_data["Wage"].copy()

    for param in param_list:
        # y_axis = sorted_data[param] / (sorted_data["MINS"] / 90)
        y_axis = sorted_data[param] / sorted_data["GP"]
        sal_group, param_group = divide_to_salary_groups(x_axis, y_axis)

        plt.plot(sal_group, param_group, label=param)
        correlation_dict_pd[param] = [
            (x_axis.cov(y_axis) / math.sqrt(x_axis.var() * y_axis.var())), param]
        correlation_dict_made[param] = [
            calc_Pearson_correlation(x_axis, y_axis), param]

    # Plot Graphs
    plt.suptitle("MLS performance to salary from 2007-2017")
    plt.title("Normalize to highest personal stat")
    plt.xlabel("Base Salary + Guaranteed Compensation")
    plt.ylabel("Performance per Games Played over one seson")
    plt.legend(loc="upper left", ncol=3)
    plt.show()

    # Plot Pearson Scatter
    plt.suptitle("MLS performance to salary Pearson-Correlation")
    plt.title("From 2007 to 2017")
    x_axis = list(range(1, len(param_list) + 1))
    # y_data = sorted([y for y in correlation_dict_made.values()], key=lambda item: item[0],reverse=True)
    y_data = sorted([y for y in correlation_dict_pd.values()],
                    key=lambda item: item[0], reverse=True)
    y_axis = [elem[0] for elem in y_data]
    labels = [elem[1] for elem in y_data]
    for index in range(len(x_axis)):
        plt.scatter(x_axis[index], y_axis[index], label=labels[index])
    plt.xlabel("Rank")
    plt.ylabel("Pearson Correlation")
    plt.legend()
    plt.show()
    return correlation_dict_pd

def get_salary_groups(salaries: pd.Series, rate: int = 1.3) -> list:
    """Obtains the salary groups

    Args:
        salaries (pd.Series): DataFrame containing Salary Data
        rate (int, optional): Rate. Defaults to 1.3.

    Returns:
        list: List containing salary groups.
    """
    base = salaries.loc[0]
    index_list = list()
    cur_index = 0
    for x in salaries.index:
        salary = salaries.loc[x]
        if salary > base * rate:
            base = salary
            index_list.append(cur_index)
        cur_index += 1
    index_list.append(salaries.size)
    return index_list

def divide_to_salary_groups(salaries: pd.Series, parameter: pd.Series):
    """Divides the dataframe to the salary groups

    Args:
        salaries (pd.Series): Salary DataFrame
        parameter (pd.Series): Param to divide the DataFrame

    Returns:
        list,list: Lists containing both axis
    """
    salary_group_index = get_salary_groups(salaries)
    salary_axis = list()
    param_axis = list()
    low = 0
    for index in salary_group_index:
        high = index
        norm = high - low
        salary_data = salaries.loc[low:high]
        parameter_data = parameter.loc[low: high]
        salary_axis.append(salary_data.sum() / norm)
        param_axis.append(parameter_data.sum() / norm)
        low = high
    return salary_axis, param_axis

def calc_Pearson_correlation(data_list1: pd.Series, data_list2: pd.Series) -> float:
    """Calculates pearson correlation between the two Series

    Args:
        data_list1 (pd.Series): Series containing the first  parameter
        data_list2 (pd.Series): Series containing the second parameter

    Returns:
        float: Pearson correlation of both parameters
    """
    covariance = cov(data_list1, data_list2)
    sqrt_var = math.sqrt(var(data_list1) * var(data_list2))
    return covariance / sqrt_var


def cov(data_list1: pd.Series, data_list2: pd.Series) -> float:
    """Calculates covariance of between the two Series

    Args:
        data_list1 (pd.Series): Serie containing the first parameter
        data_list2 (pd.Series): Serie containing the second parameter

    Returns:
        float: covariance of both parameters
    """
    m1 = data_list1.sum() / data_list1.size
    m2 = data_list2.sum() / data_list2.size
    m3 = 0
    # for index in range(len(data_list1)):
    for index in data_list1.index:
        m3 += data_list1.loc[index] * data_list2.loc[index]
    m3 = m3 / data_list1.size
    result = m3 - (m1 * m2)
    return result


def var(data_list: pd.DataFrame) -> float:
    """Calculates de variance of the data

    Args:
        data_list (pd.DataFrame): DataFrame containing the data     

    Returns:
        float: Variance of the DataFrame
    """
    b = (data_list.sum() / data_list.size) ** 2
    a = 0
    for elem in data_list:
        a += elem ** 2
    a = a / data_list.size
    return a - b


def wins_by_parameter(table_data: pd.DataFrame, param_list: list, win_param: str = "W", group_factor=1.4, group_index=1) -> dict:
    """plots the number of wins according to a parameter

    Args:
        table_data (pd.DataFrame): DataFrame containing the teams data and sum of parameters per team
        param_list (list): list of parameters to plot
        win_param (str, optional): Win parameter to plot against. Defaults to "W".
        group_factor (float, optional): Factor to group. Defaults to 1.4.
        group_index (int, optional): Index to group. Defaults to 1.

    Returns:
        dict: Dictionary containing team performances
    """
    team_performances_dict = dict()
    for param in param_list:
        x_axis = list()
        y_axis = list()
        for year in range(2007, 2018):
            y_axis_yearly = table_data[table_data.Year == year][win_param]
            x_axis_param_yearly = table_data[table_data.Year == year][param]
            # zero_index_yearly = y_axis_yearly[y_axis_yearly.W == 0]
            # y_axis_yearly = y_axis_yearly[y_axis_yearly.W > 0]

            plt.scatter(x_axis_param_yearly, y_axis_yearly, label=year)
            x_axis += x_axis_param_yearly.tolist()
            y_axis += y_axis_yearly.tolist()
        # Plot scatter of all data
        plt.title(f"MLS {win_param} by {param} from 2007 to 2017")
        plt.xlabel(param)
        plt.ylabel(win_param)
        plt.legend()
        plt.show()

        # Plot Scatter of data groups
        x_group_axis, y_group_axis = into_groups(
            x_axis, y_axis, group_factor, group_index)
        if group_index == 1:
            x_group_axis, y_group_axis = sort_to_plot(
                x_group_axis, y_group_axis)

        plt.title(f"MLS {param} to {win_param} correlation from 2007 to 2017")
        plt.xlabel(f"total {param}")
        plt.ylabel(f"Yearly {win_param}")
        plt.plot(x_group_axis, y_group_axis)
        plt.show()
        pearson = calc_Pearson_correlation(
            pd.Series(x_axis), pd.Series(y_axis))
        team_performances_dict[param] = [
            x_group_axis, y_group_axis, pearson, param]
    for param in param_list:
        plt.title(f"MLS {param} to {win_param} correlation from 2007 to 2017")
        plt.xlabel(f"total {param}")
        plt.ylabel(f"Yearly {win_param}")
        data = team_performances_dict[param][0]
        x_axis = [x / max(data) for x in data]
        y_axis = team_performances_dict[param][1]
        plt.plot(x_axis, y_axis,
                 label=f"{param} corr: {team_performances_dict[param][2]}")
    plt.legend()
    plt.show()
    print(team_performances_dict)
    # plot pearson correlation
    plt.title(
        f"MLS game-aspect to {win_param} pearson-correlation : 2007-2017")
    plt.xlabel("Rank")
    plt.ylabel("Pearson Correlation")
    x_axis = list(range(1, len(param_list) + 1))
    y_data = sorted([x for x in team_performances_dict.values()],
                    key=lambda x: x[2], reverse=True)

    y_axis = [elem[2] for elem in y_data]
    labels = [elem[3] for elem in y_data]

    for index in range(len(x_axis)):
        plt.scatter(x_axis[index], y_axis[index], label=labels[index])
    plt.legend()
    plt.show()
    return team_performances_dict

def search_k_means_communities(teams, team_performance, salary_metrics, parameter, limit=ANALYSIS_BAR, x=AVG, y=MED, w=RAT):
    """Search for k mean communities between the team_performance, salary metrics against a parameter

    Args:
        teams (list): list of teams
        team_performance (dict): Dictionary of teams performance
        salary_metrics (dict): Dictionary of salary metrics
        parameter (str): Param to find the communities
        limit (int, optional): Limit for the analysis to stop. Defaults to ANALYSIS_BAR.
        x (str, optional): Parameter for x-axis. Defaults to AVG.
        y (str, optional): Parameter for y-axis. Defaults to MED.
        w (str, optional): Parameter for z-axis. Defaults to RAT.

    Returns:
        _type_: _description_
    """
    output = create_salary_communities(
        teams, team_performance, salary_metrics, parameter, x, y, w)
    while max(output[2]) <= limit:
        print(f"max: {max(output[2])}, min: {min(output[2])}")
        output = create_salary_communities(
            teams, team_performance, salary_metrics, "W")
    print(f"max: {max(output[2])}, min: {min(output[2])}")
    return output


def create_salary_communities(team_list, team_performances, salary_metrics, parameter, x=AVG, y=MED, w=RAT):
    """Creates salary communities from the data

    Args:
        team_list (list): List of teams
        team_performances (dict): Dictionary containing team performances data
        salary_metrics (dict): Dictionary containing the salary metrics
        parameter (str): Parameter to find community against
        x (str, optional): Parameter for x-axis. Defaults to AVG.
        y (str, optional): Parameter for y-axis. Defaults to MED.
        w (str, optional): Parameter for z-axis. Defaults to RAT.

    Returns:
        list,list,list,list,list: Lists of the Data of the communities
    """

    x_axis = list()
    y_axis = list()
    z_axis = list()
    w_axis = list()
    for team in team_list:
        for year in range(2007, 2018):
            try:
                w_axis.append(salary_metrics[(team, year)][2][w])
                x_axis.append(salary_metrics[(team, year)][2][x])
                y_axis.append(salary_metrics[(team, year)][2][y])
                z_axis.append(team_performances[year][team][parameter])
            except KeyError:
                # print(f"No info for ({team},{year})")
                1+1
    # PLOT 3D AMS DISTRIBUTION
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_axis, y_axis, z_axis)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(parameter)
    plt.title(f"MLS team's AMS({parameter}) distribution in 2007-2017")
    plt.show()
    # PLOT 2D BALANCE-WINS DISTRIBUTION
    plt.scatter(z_axis, w_axis)
    plt.title(f"MLS team's {parameter} to {w} : 2007-2017")
    plt.ylabel(f"{w}")
    plt.xlabel(parameter)
    plt.show()
    # PLOT 3D AMS COMMUNITIES
    norm = normalize_for_k_means(x_axis, y_axis, z_axis)
    communities = k_means((x_axis, y_axis, z_axis),
                          norm, K, AMS_DIM, K_MEAN_STEPS)
    x_axis, y_axis, z_axis, size_axis, balance_axis = centroids_to_points(
        communities)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_axis, y_axis, z_axis, s=size_axis,
                 color="orange", edgecolor="black")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(parameter)
    plt.suptitle(f"MLS AMS({parameter}) communities : 2007-2017")
    plt.title(f"K: {K}, steps:{K_MEAN_STEPS}")
    plt.show()
    # PLOT 2D BALANCE-WINS DISTRIBUTION
    for index in range(len(x_axis)):
        if balance_axis[index] != -1 and z_axis != 0:
            label = f"BS: {arrange_decimal_point(balance_axis[index])}, W: {arrange_decimal_point(z_axis[index])}"
            plt.scatter(z_axis[index], balance_axis[index],
                        s=size_axis[index], label=label)
    plt.title(f"MLS AMS communities {parameter} to {w} : 2007-2017")
    plt.xlabel(parameter)
    plt.ylabel(f"{w}")
    plt.legend()
    plt.show()
    return x_axis, y_axis, z_axis, size_axis, balance_axis

def centroids_to_points(communities):
    """Obtains the centroids and transforms it to points

    Args:
        communities (dict): Dictionary containing communities

    Returns:
        list,list,list,list,list: list containing data of the centroids for the plot
    """
    x_axis = list()
    y_axis = list()
    z_axis = list()
    size_axis = list()
    balance_axis = list()
    centroids = list(communities.keys())
    for index in range(len(centroids)):
        x_axis.append(centroids[index].place[0])
        y_axis.append(centroids[index].place[1])
        z_axis.append(centroids[index].place[2])
        size_axis.append(10*len(communities[centroids[index]]))
        if y_axis[-1] != 0:
            balance_axis.append(x_axis[-1] / y_axis[-1])
        else:
            balance_axis.append(-1)
    return x_axis, y_axis, z_axis, size_axis, balance_axis

def normalize_for_k_means(x_axis, y_axis, z_axis):
    """Normalizes points for the k-means

    Args:
        x_axis (list): list of x-axis points
        y_axis (list): list of y-axis points
        z_axis (list): list of z-axis points

    Returns:
        list, list, list: list of normalized points
    """
    x_norm = max(x_axis) - min(x_axis)
    y_norm = max(y_axis) - min(y_axis)
    z_norm = max(z_axis) - min(z_axis)

    norm_x = max(x_axis)
    x_axis = [x / norm_x for x in x_axis]
    norm_y = max(y_axis)
    y_axis = [y / norm_y for y in y_axis]
    norm_z = max(z_axis)
    z_axis = [z / norm_z for z in z_axis]
    return x_norm, y_norm, z_norm

def k_means(data_points, norm, k, dim, stop=5):
    """k-means algorithms

    Args:
        data_points (tuple): tuple containing the data of the points
        norm (tuple): tuple containing the norm
        k (int): number of groups for the algorithm
        dim (int): dimension of the groups
        stop (int, optional): number of steps for the algorithm. Defaults to 5.

    Returns:
        dict: dictionary containing communities
    """
    limits = calc_limits(data_points, dim)
    centroids = create_centroids(limits, k)
    points = get_points(data_points, dim)
    start_communities = calc_communities(points, centroids, norm)
    final_communities = k_means_help(points, centroids, start_communities, dim,
                                     stop, norm)
    return final_communities


def k_means_help(points, centroids, communities, dim, stop, norm):
    """Helper function for the k-means algorithm

    Args:
        points (list): list containing the points
        centroids (list): list containing centroids
        communities (dict): Dictionary containing communities
        dim (int): Dimension of the groups
        stop (int): Number of steps for the algorithm to execute
        norm (tuple): tuple containing the norm

    Returns:
        dict: Dictionary containing communities
    """
    cycles = 0
    centroids = adjust_centroids(centroids, communities, dim)
    new_communities = calc_communities(points, centroids, norm)
    while cycles < stop:
        if same_communities(communities, new_communities):
            return communities
        else:
            communities = new_communities
            centroids = adjust_centroids(centroids, communities, dim)
            cycles += 1
    return communities


def same_communities(community1, community2):
    """Returns True if the two communities are the same

    Args:
        community1 (dict): Dictionary of communities to compare
        community2 (dict): Dictionary of communities to compare

    Returns:
        Boolean: True if the two communities are the same. False otherwise.
    """
    for community in community1.values():
        if community not in community2.values():
            return False
    return True


def adjust_centroids(centroids, communities, dim):
    """Adjusts the centroids of the communities

    Args:
        centroids (list): list of centroids
        communities (dict): dictionary containing communities
        dim (int): dimension of the graph

    Returns:
        list: list of new centroids
    """
    new_centroids = list()
    for centroid in centroids:
        centroid = adjust_centroid(communities[centroid], dim)
        new_centroids.append(centroid)
    return new_centroids


def adjust_centroid(points, dim):
    """Adjusts the centroid of the points

    Args:
        points (list): List of points
        dim (int): Dimension of the plot

    Returns:
        Centroid: New Centroid of the points
    """
    new_dim = dict()
    for d in range(dim):
        new_dim[d] = list()
    for point in points:
        for d in range(dim):
            new_dim[d].append(point[d])
    new_centroid = Centroid(new_dim)
    for d in range(dim):
        if len(points) > 0:
            new_centroid.place[d] = sum(new_centroid.place[d]) / len(points)
        else:
            new_centroid.place[d] = 0
    return new_centroid


def calc_communities(points, centroids, norm):
    """Calculates the new communities

    Args:
        points (list): List of points
        centroids (list): List of centroids
        norm (int): Norm

    Returns:
        dict: Dictionary of comunities
    """
    communities = dict()
    for centroid in centroids:
        communities[centroid] = list()
    for point in points:
        closet_centroid = find_closest(point, centroids, norm)
        communities[closet_centroid].append(point)
    return communities


def find_closest(point, centroids, norm):
    """Find the closests centroid of the points

    Args:
        point (list): list of points
        centroids (list): List of centroids
        norm (norm): Norm

    Returns:
        Centroid: Closest Centroid of the points
    """
    closest_centroid = centroids[0]
    closest_dist = calc_dist(point, centroids[0].place, norm)
    for centroid in centroids:
        dist = calc_dist(point, centroid.place, norm)
        if dist < closest_dist:
            closest_dist = dist
            closest_centroid = centroid
    return closest_centroid


def calc_dist(point, centroid, norm):
    """Calculates distance from point to centroid

    Args:
        point (tuple): point
        centroid (tuple): point of centroid
        norm (norm): norm

    Returns:
        int: Eucledian distance from the point to the centroid
    """
    dist = 0
    for index in range(len(point)):
        dist += ((point[index] - centroid[index]) / norm[index]) ** 2
    dist = dist ** 0.5
    return dist


def get_points(data_points, dim):
    """Gets the points from the data points

    Args:
        data_points (tuple): Tuple of data points
        dim (int): dimension of the plot

    Returns:
        list: list of points
    """
    points = list()
    for index in range(len(data_points[0])):
        point = list()
        for d in range(dim):
            point.append(data_points[d][index])
        points.append(point)
    return points


def calc_limits(data_points, dim):
    """Calculates limits of the data points

    Args:
        data_points (tuple): Tuple of the data points
        dim (int): Dimension of the plot

    Returns:
        list: List of the limits
    """
    limits = list()
    for d in range(dim):
        high = max(data_points[d])
        low = min(data_points[d])
        limits.append([low, high])
    return limits


def create_centroids(limits, k):
    """Creates centroids from the limits

    Args:
        limits (list): List of limits
        k (int): Number of groups for the algorithm

    Returns:
        list: List of centroids
    """
    centroids = list()
    for i in range(k):
        c = list()
        for d in range(len(limits)):
            c.append(random.uniform(limits[d][0], limits[d][1]))
        centroid = Centroid(c)
        centroids.append(centroid)
    return centroids

def calculate_team_salary_metrics(players_dictionary: pd.DataFrame, team_list: list) -> dict:
    """Calculates the team salary metrics from a DataFrame

    Args:
        players_dictionary (pd.DataFrame): DataFrame containing the players data
        team_list (list): List of teams

    Returns:
        dict: Dictionary containing team salary metrics
    """
    salary_metrics = dict()
    
    for x in players_dictionary.index:
        team = players_dictionary.loc[x, "Club"]
        if team not in salary_metrics:
            salary_metrics[team] = [[], 0, dict()]
        salary_metrics[team][0].append(players_dictionary.loc[x, "Wage"])
        salary_metrics[team][1] += 1
    for team in team_list:
        element = salary_metrics[team]
        element[0] = sorted(element[0], key=lambda x: x)
        avg = int(calc_avg(element[0]))
        median = int(calc_median(element[0]))
        element[2][AVG] = avg
        element[2][MED] = median
        element[2][RAT] = avg / median
        element[2][NORM_RAT] = math.log(avg / median, 1.7)
        element[2][DIST] = max(element[0]) - min(element[0])
        element[2][VAR] = var(pd.DataFrame(element[0]))
        element[2][GINI] = gini(element[0])
    return salary_metrics

def calculate_team_salary_metrics_yearly(players_dictionary: pd.DataFrame, team_list: list) -> dict:
    """Calculates the team salary metrics per year

    Args:
        players_dictionary (pd.DataFrame): DataFrame containing players data
        team_list (list): List of teams

    Returns:
        dict: Dictionary containing team salary metrics per year
    """
    salary_metrics = dict()
    
    for x in players_dictionary.index:
        team = players_dictionary.loc[x, "Club"]
        year = players_dictionary.loc[x, "Year"]
        if (team, year) not in salary_metrics:
            salary_metrics[(team, year)] = [[], 0, dict()]
        salary_metrics[(team, year)][0].append(
            players_dictionary.loc[x, "Wage"])
        salary_metrics[(team, year)][1] += 1
    for team in team_list:
        for year in range(2007, 2018):
            try:
                element = salary_metrics[(team, year)]
                element[0] = sorted(element[0], key=lambda x: x)
                avg = int(calc_avg(element[0]))
                median = int(calc_median(element[0]))
                element[2][AVG] = avg
                element[2][MED] = median
                element[2][RAT] = avg / median
                element[2][NORM_RAT] = math.log(avg / median, 1.7)
                element[2][DIST] = max(element[0]) - min(element[0])
                element[2][VAR] = var(pd.DataFrame(element[0]))
                element[2][GINI] = gini(element[0])
            except KeyError:
                # print(f"No infor for ({team},{year})")
                1+1
    return salary_metrics

def arrange_decimal_point(num, d=3):
    """Arranges decimal point from a number

    Args:
        num (int | float): Number to adjust
        d (int, optional): Factor to adjust the decimal point. Defaults to 3.

    Returns:
        float: Number after decimal arrangement
    """
    factor = 10 ** d
    return int(num * factor) / factor

def calc_avg(lst):
    """Calculates average from the list

    Args:
        lst (list): list of numbers

    Returns:
        float: Average of the numbers from the list
    """
    return sum(lst) / len(lst)

def calc_median(lst):
    """Calculates median from list

    Args:
        lst (list): list of numbers

    Returns:
        float: median of the list of numbers
    """
    return lst[int(len(lst) / 2)]

def gini(data_list):
    """Calculates the gini criterion from the list

    Args:
        data_list (list): list of numbers

    Returns:
        float: gini criterion
    """
    n = len(data_list)
    s = sum(data_list)
    total_dist = 0
    for i in data_list:
        for j in data_list:
            total_dist += abs(i - j)
    g = total_dist / (2*n*s)
    return g

def create_team_performance_dict(players_data, team_dataset: pd.DataFrame, team_list, parameter_list=DEFAULT_STATS):
    """Creates dictionary of the team performance

    Args:
        players_data (pd.DataFrame): DataFrame containing players data
        team_dataset (pd.DataFrame): DataFrame containing teams table data
        team_list (list): List of teams
        parameter_list (list, optional): list of parameters. Defaults to DEFAULT_STATS.

    Returns:
        dict: Dictionary of team performances
    """
    teams_performances = dict()
    for year in range(2007, 2018):
        year_performance = dict()
        for team in team_list:
            team_data = dict()
            # team_data["W"] = count_wins(team_dataset, team, year)
            team_data["W"] = team_dataset[(team_dataset.Club == team) & (
                team_dataset.Year == year)]["W"].sum()
            team_data["Pts"] = team_dataset[(team_dataset.Club == team) & (
                team_dataset.Year == year)]["Pts"].sum()
            for parameter in parameter_list:
                team_data[parameter] = sum_team_parameter_per_year(
                    players_data, parameter, year, team)
            year_performance[team] = team_data
        teams_performances[year] = year_performance
    return teams_performances

def sum_team_parameter_per_year(players_dictionary: pd.DataFrame, parameter, year, team):
    """Sums parameter of a team per year

    Args:
        players_dictionary (pd.DataFrame): Dictionary containing players stats
        parameter (str): Parameter to add
        year (int): Year to evaluate
        team (str): Team abreviation

    Returns:
        int: Sum of the parameter
    """
    param_sum = players_dictionary[(players_dictionary.Year == year) & (
        players_dictionary.Club == team)][parameter].sum()

    return param_sum

def into_groups(x_axis, y_axis, group_factor, group_index=0):
    """Creates groups from the data

    Args:
        x_axis (list): List of points from x-axis
        y_axis (list): List of points from y-axis
        group_factor (float): Value for the group factor
        group_index (int, optional): Value for the group index. Defaults to 0.

    Returns:
        list,list: list of groups for every axis
    """
    x_group_axis = list()
    y_group_axis = list()
    to_sort_list = list()
    for index in range(len(x_axis)):
        to_sort_list.append([x_axis[index], y_axis[index]])
    sorted_list = sorted(to_sort_list, key=lambda elem: elem[group_index])
    base = sorted_list[0][group_index]
    count = 0
    x_temp = 0
    y_temp = 0
    for item in sorted_list:
        if item[group_index] <= base * group_factor:
            x_temp = (x_temp * count + item[0]) / (count + 1)
            y_temp = (y_temp * count + item[1]) / (count + 1)
        else:
            x_group_axis.append(x_temp)
            y_group_axis.append(y_temp)
            base = item[group_index]
            x_temp = item[0]
            y_temp = item[1]
            count = 0
        count += 1
    return x_group_axis, y_group_axis

def sort_to_plot(base_x_axis, base_y_axis):
    """Sorts points to plot them

    Args:
        base_x_axis (list): list of x axis
        base_y_axis (list): list of y axis

    Returns:
        list, list: list of sorted points for each axis
    """
    x_axis = list()
    y_axis = list()
    to_sort_list = list()
    for index in range(len(base_x_axis)):
        to_sort_list.append([base_x_axis[index], base_y_axis[index]])
    sorted_list = sorted(to_sort_list, key=lambda elem: elem[0])
    for index in range(len(base_x_axis)):
        x_axis.append(sorted_list[index][0])
        y_axis.append(sorted_list[index][1])
    return x_axis, y_axis

def train_evaluation(team_list, team_performance, yearly_salary):
    """Creates plot from data to evaluate predictions according to the number of wins and balance-score from the data we have

    Args:
        team_list (list): List of teams
        team_performance (dict): Dictionary of team performances
        yearly_salary (dict): Dictionary of yearly salaries per team
    """
    column = range(0, 40, 4)
    golden_ratio = [[3.2, 6.0]]
    bad_zone = 1.9  # 6?
    search_score = 0
    champion_find = 0
    loser_find = 0
    count = 0
    top_x_list = list()
    top_y_list = list()
    bottom_x_list = list()
    bottom_y_list = list()
    top_flag = False
    bottom_flag = False
    east_flag = False
    west_flag = False
    for year in range(2007, 2017+1):
        year_wins_list = [team_performance[year][team]["W"]
                          for team in team_list]
        top, bottom = get_top_bottom(year_wins_list)
        for team in team_list:
            try:
                count += 1
                color = 'orange'
                edge = 'white'
                x = yearly_salary[(team, year)][2][RAT]
                y = team_performance[year][team]["W"]
                if REG_SEASON_WEST[year] == team:
                    color = 'blue'
                    edge = 'black'
                    if get_search_score(x, golden_ratio) > 0:
                        champion_find += 5
                    if not west_flag:
                        plt.scatter(x, y, color=color, edgecolors=edge,
                                    label="west champion")
                        west_flag = True
                    plt.scatter(x, y, color=color, edgecolors=edge)
                elif REG_SEASON_EAST[year] == team:
                    color = 'purple'
                    edge = 'black'
                    if get_search_score(x, golden_ratio) > 0:
                        champion_find += 5
                    if not east_flag:
                        plt.scatter(x, y, color=color, edgecolors=edge,
                                    label="east champion")
                        east_flag = True
                    plt.scatter(x, y, color=color, edgecolors=edge)
                elif y >= top:
                    search_score += get_search_score(x, golden_ratio)
                    color = 'green'
                    edge = 'black'
                    top_x_list.append(x)
                    top_y_list.append(y)
                    if not top_flag:
                        plt.scatter(x, y, color=color,
                                    edgecolors=edge, label="best 10%")
                        top_flag = True
                    plt.scatter(x, y, color=color, edgecolors=edge)
                elif y <= bottom:
                    color = 'red'
                    edge = 'black'
                    bottom_x_list.append(x)
                    bottom_y_list.append(y)
                    if x > bad_zone:
                        loser_find += 1
                    if not bottom_flag:
                        plt.scatter(x, y, color=color,
                                    edgecolors=edge, label="worst 30%")
                        bottom_flag = True
                    plt.scatter(x, y, color=color, edgecolors=edge)
                else:
                    plt.scatter(x, y, color=color, edgecolors=edge)
            except KeyError:
                1+1
    # PLOTS
    plt.plot([3.2 ] * 10, column, label="BS: 3.2", color="green")
    plt.plot([6.00] * 10, column, label="BS: 6.00", color="green")
    plt.plot([bad_zone] * 10, column, label=f"BS: {bad_zone}", color="red")
    plt.suptitle(f"MLS wins to Balance-Score : {2007}-{2017}")
    plt.title("Evaluation plot")
    plt.xlabel("Balance Score")
    plt.ylabel("wins")
    plt.legend()
    plt.show()
    print(f"Recall: {search_score}/{count * 0.1} of best 10%")
    print(f"champions: {champion_find}%")
    print(f"Recall: {loser_find}/{count * 0.3} of worst 30%")
    print(st.spearmanr(top_x_list, top_y_list))
    print(st.pearsonr(top_x_list, top_y_list))
    print(st.spearmanr(bottom_x_list, bottom_y_list))
    print(st.pearsonr(bottom_x_list, bottom_y_list))
    
def evaluation(salary_metric, team_list):
    """Evaluates our predictions from new data
        We need to adjust the lower, upper bound and evaluation win data for each year

    Args:
        salary_metric (dict): Dictionary containing teams salary metrics
        team_list (list): List of teams 
    """
    
    column = range(0, 33, 3)
    top_flag = False
    bottom_flag = False
    golden_ratio = [[3.2, 6.0]]
    bad_zone = 1.9 #beyond this, you're usually bad team
    search_score = 0
    champion_find = 0
    loser_find = 0
    for team in team_list:
        try:
            x = salary_metric[team][2][RAT]
            y = EVALUATION_WINS_2020[team]
            if x > 0:
                edge = 'white'
                color = 'orange'
                if team == 'CLB':
                    edge = 'black'
                    color = 'blue'
                    plt.scatter(x, y, color=color, edgecolors=edge, label='Champion')
                    if get_search_score(x, golden_ratio) > 0:
                        champion_find += 100
                        search_score += 1
                elif y >= 12: #upper
                    search_score += get_search_score(x, golden_ratio)
                    color = 'green'
                    edge = 'black'
                    if not top_flag:
                        plt.scatter(x, y, color=color, edgecolors=edge, label="top 10%")
                        top_flag = True
                    else:
                        plt.scatter(x, y, color=color, edgecolors=edge)
                elif y <= 8: #lower
                    color = 'red'
                    edge = 'black'
                    if x < bad_zone:
                        loser_find += 1
                    if not bottom_flag:
                        plt.scatter(x, y, color=color, edgecolors=edge, label="bottom 30%")
                        bottom_flag = True
                    else:
                        plt.scatter(x, y, color=color, edgecolors=edge)
                else:
                    plt.scatter(x, y, color=color, edgecolors=edge)
        except KeyError:
            1+1
    # PLOTS
    plt.plot([3.2] * 11, column, label="BS: 3.2", color="green")
    plt.plot([6.0] * 11, column, label="BS: 6.0", color="green")
    plt.plot([1.9] * 11, column, label="BS: 1.9", color="red")
    plt.suptitle(f"MLS wins to Balance-Score : {2020}")
    plt.title("Evaluation plot")
    plt.xlabel("Balance Score")
    plt.ylabel("wins")
    plt.legend()
    plt.show()
    print(f"Recall: {search_score}/2 of best 10%")
    print(f"champions: {champion_find}%")
    print(f"Recall: {loser_find}/7 under 10 wins")


def get_top_bottom(wins_list):
    """Gets the top and bottom from the list of wins

    Args:
        wins_list (list): List of wins of teams

    Returns:
        int,int: top and bottom of the list
    """
    sorted_list = sorted(wins_list, key=lambda elem: elem)
    return sorted_list[-3], sorted_list[8]


def get_search_score(x, golden_ratio):
    """Gets if the number is inside the golen ratio

    Args:
        x (float): Number to evaluate
        golden_ratio (list): list containing parameters of golden ratio

    Returns:
        int: 1 if the number is between the golden ratio, 0 otherwise
    """
    for interval in golden_ratio:
        if interval[0] < x < interval[1]:
            return 1
    return 0


def get_players_data() -> pd.DataFrame:
    """Gets the players data from the csv file. Removes unnecesary columns and normalize names.

    Returns:
        pd.DataFrame: DataFrame of the players data
    """
    data = load_data(PLAYER_DATASET)
    data = preprocess_players(data)
    col = [x for x in list(data.columns)
           if x not in COLUMNS_PLAYERS + DEFAULT_STATS]
    data.drop(col, axis=1, inplace=True)
    return data

def get_table_data() -> pd.DataFrame:
    """Gets tha teams table data from the csv files. Process the data to obtain the desired data.

    Returns:
        pd.DataFrame: DataFrame of the teams data.
    """
    data = load_data(TEAM_DATASET)
    data = preprocess_table(data)
    return data


if __name__ == "__main__":

    # #We start analyzing the data,
    #  generating charts
    # for year in range(2007,2018):
    #     for position in STATS_DICT.keys():
    #         for param in STATS_DICT.get(position):
    #             data_per_position = limit_by_position(players, position)
    #             visualization(data_per_position,param,year,position)

    table = get_table_data()
    salary = get_full_salaries()
    players = get_players_data()
    # Players and Salary dataset
    dataset = pd.merge(salary, players, on=["Player", "Year", "Club"])
    
    ##Step one
    plot_salary_distribution(dataset,"Wage")    #Wage = base_salary + guaranteed_compensation

    ##Step two and three
    tm_pf_dict = plot_salary_param_distribution(dataset,DEFAULT_STATS)
    

    
    team_performances = players[["Club", "Year", "G", "A", "SOG", "GWG",
                                 "GWA", "G/90min", "A/90min"]].groupby(["Club", "Year"]).sum()
    team_performances = pd.merge(team_performances, table, on=["Club", "Year"])
    tm_pf_dict1 = wins_by_parameter(team_performances,DEFAULT_STATS)
    tm_pf_dict2 = wins_by_parameter(team_performances,DEFAULT_STATS,"Num_Championships")
    
    # EVALUATION
    team_list = dataset["Club"].unique().tolist()

    team_performance_dict = create_team_performance_dict(players, table, team_list)
    salary_metrics_dict = calculate_team_salary_metrics_yearly(dataset, team_list)
    
    salary_metrics_1 = calculate_team_salary_metrics(dataset,team_list)
    search_k_means_communities(team_list,team_performance_dict,salary_metrics_dict,"W")
    train_evaluation(team_list, team_performance_dict,salary_metrics_dict) 
    evaluation(salary_metrics_1,team_list)
      
    
