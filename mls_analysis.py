import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import math

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

MIN_MINS_PLAYED = 500
MIN_GAMES_PLAYED = 5

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

DEFAULT_STATS = ["G", "A", "SOG", "FC", "G/90min", "A/90min","GWG","GWA"]

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
    for year in range(2007,2018):
        data = get_year_salary(year)
        data["Year"] = year
        res = pd.concat([res,data],ignore_index=True)
    res = preprocess_salaries(res)
    res.dropna(axis=0,inplace=True)
    return res

def load_data(csv_file_path: str) -> pd.DataFrame:
    """ Loads the csv file from the path provided."""
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
    dataset['Wage']   = dataset['base_salary'] + dataset['guaranteed_compensation']
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

    dataset = dataset[dataset.MINS >= MIN_MINS_PLAYED]
    dataset = dataset[dataset.GP   >= MIN_GAMES_PLAYED]
    
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
        if dataset.loc[x,"Champion"]:
            dataset.loc[x,"Num_Championships"] = 1
        else:
            dataset.loc[x,"Num_Championships"] = 0
    dataset = normalize_team_names(dataset)
    
    dataset.drop(columns=["SW","SL","D","Qualification","Conference","Head-to-head","PPG"],inplace=True)
    
    if club is not None:
        dataset = dataset[dataset.Club == club]
    
    return dataset

def plot_salary_distribution(salary_dataset: pd.DataFrame,param = "base_salary"):
    salary_dataset.sort_values(by=[param],inplace=True,ignore_index=True)
    salary_group_index = get_salary_groups(salary_dataset[param],1.5)
    
    salary_axis = list()
    amount_axis = list()
    
    low = 0
    for index in salary_group_index:
        high = index
        norm = high - low
        amount_axis.append(norm)
        salary_axis.append(sum(salary_dataset.loc[low:high,param]) / norm)
        low = high
    plt.title("MLS Salary Distribution 2007 - 2017")
    plt.xlabel(param)
    plt.ylabel("Amount of players")
    plt.plot(salary_axis,amount_axis)
    plt.show()
    # print(salary_axis)
    # print(len(amount_axis),(salary_axis))
    
    
def plot_salary_param_distribution(salary_dataset: pd.DataFrame, param_list: list):
    sorted_data = salary_dataset.sort_values(by=["Wage"],ignore_index=True)
    correlation_dict_pd = dict()
    correlation_dict_made = dict()
    x_axis = sorted_data["Wage"].copy()
    
    for param in param_list:
        y_axis = sorted_data[param] / sorted_data["MINS"]
        sal_group, param_group = divide_to_salary_groups(x_axis,y_axis)
        
        plt.plot(sal_group, param_group, label=param)
        correlation_dict_pd[param] = [(x_axis.cov(y_axis) / math.sqrt(x_axis.var() * y_axis.var())),param]
        correlation_dict_made[param] = [calc_Pearson_correlation(x_axis,y_axis),param]
        
    #Plot Graphs
    plt.suptitle("MLS performance to salary from 2007-2017")
    plt.title("Normalize to highest personal stat")
    plt.xlabel("Base Salary + Guaranteed Compensation")
    plt.ylabel("Performance per minute over one seson")
    plt.legend(loc="upper left", ncol=3)
    plt.show()
    
    #Plot Pearson Scatter
    plt.suptitle("MLS performance to salary Pearson-Correlation")
    plt.title("From 2007 to 2017")
    x_axis = list(range(1,len(param_list) + 1))
    # y_data = sorted([y for y in correlation_dict_made.values()], key=lambda item: item[0],reverse=True)
    y_data = sorted([y for y in correlation_dict_pd.values()], key=lambda item: item[0], reverse= True)
    y_axis = [elem[0] for elem in y_data]
    labels = [elem[1] for elem in y_data]
    for index in range(len(x_axis)):
        plt.scatter(x_axis[index],y_axis[index],label=labels[index])
    plt.xlabel("Rank")
    plt.ylabel("Pearson Correlation")
    plt.legend()
    plt.show()
    
def get_salary_groups(salaries: pd.Series, rate: int = 1.3) -> list:
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
        param_axis.append(parameter_data.sum()/ norm)
        low = high
    return salary_axis, param_axis

def calc_Pearson_correlation(data_list1: pd.Series, data_list2: pd.Series) -> float:
    covariance = cov(data_list1, data_list2)
    sqrt_var = math.sqrt(var(data_list1) * var(data_list2))
    return covariance / sqrt_var


def cov(data_list1: pd.Series, data_list2: pd.Series) -> float:
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
    b = (data_list.sum() / data_list.size) ** 2
    a = 0
    for elem in data_list:
        a += elem ** 2
    a = a / data_list.size
    return a - b

def wins_by_parameter(table_data: pd.DataFrame, param_list: list,win_param: str = "W", group_factor = 1.4, group_index = 1):
    team_performances_dict = dict()
    for param in param_list:
        x_axis = list()
        y_axis = list()
        for year in range(2007,2018):
            y_axis_yearly = table_data[table_data.Year == year][win_param]
            x_axis_param_yearly = table_data[table_data.Year == year][param]
            # zero_index_yearly = y_axis_yearly[y_axis_yearly.W == 0]
            # y_axis_yearly = y_axis_yearly[y_axis_yearly.W > 0]
            
            plt.scatter(x_axis_param_yearly,y_axis_yearly,label=year)
            x_axis += x_axis_param_yearly.tolist()
            y_axis += y_axis_yearly.tolist()
        #Plot scatter of all data
        plt.title(f"MLS {win_param} by {param} from 2007 to 2017")
        plt.xlabel(param)
        plt.ylabel(win_param)
        plt.legend()
        plt.show()
        
        #Plot Scatter of data groups
        x_group_axis, y_group_axis = into_groups(x_axis, y_axis, group_factor, group_index)
        if group_index == 1:
            x_group_axis, y_group_axis = sort_to_plot(x_group_axis,y_group_axis)
        
        plt.title(f"MLS {param} to {win_param} correlation from 2007 to 2017")
        plt.xlabel(f"total {param}")
        plt.ylabel(f"Yearly {win_param}")
        plt.plot(x_group_axis,y_group_axis)
        plt.show()
        pearson = calc_Pearson_correlation(pd.Series(x_axis),pd.Series(y_axis))
        team_performances_dict[param] = [x_group_axis,y_group_axis,pearson,param]
    for param in param_list:
        plt.title(f"MLS {param} to {win_param} correlation from 2007 to 2017")
        plt.xlabel(f"total {param}")
        plt.ylabel(f"Yearly {win_param}")
        data = team_performances_dict[param][0]
        x_axis = [x / max(data) for x in data]
        y_axis = team_performances_dict[param][1]
        plt.plot(x_axis,y_axis, label=f"{param} corr: {team_performances_dict[param][2]}")
    plt.legend()
    plt.show()
    
    #plot pearson correlation
    plt.title(f"MLS game-aspect to {win_param} pearson-correlation : 2007-2017")
    plt.xlabel("Rank")
    plt.ylabel("Pearson Correlation")
    x_axis = list(range(1, len(param_list) + 1))
    y_data = sorted([x for x in team_performances_dict.values()],key=lambda x: x[2],reverse=True)
    
    y_axis = [elem[2] for elem in y_data]
    labels = [elem[3] for elem in y_data]
    
    for index in range(len(x_axis)):
        plt.scatter(x_axis[index],y_axis[index],label=labels[index])
    plt.legend()
    plt.show()


def into_groups(x_axis, y_axis, group_factor, group_index=0):
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

def get_players_data() -> pd.DataFrame:
    data = load_data(PLAYER_DATASET)
    data = preprocess_players(data)
    col = [x for x in list(data.columns) if x not in COLUMNS_PLAYERS + DEFAULT_STATS]
    data.drop(col,axis=1, inplace=True)
    return data

def get_table_data() -> pd.DataFrame:
    data = load_data(TEAM_DATASET)
    data = preprocess_table(data)
    return data

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

    table = get_table_data()
    salary = get_full_salaries()
    players = get_players_data()
    dataset = pd.merge(salary,players,on=["Player","Year","Club"])
    # plot_salary_distribution(dataset,"Wage")
    # plot_salary_param_distribution(dataset,["G","A","SOG","GWG","GWA"])
    # dataset["Wage"].plot.hist(by=["Year"])
    # plt.show()
    
    param_list= ["G","A","SOG","G/90min","A/90min"]
    team_performances = players[["Club","Year","G","A","SOG","GWG","GWA","G/90min","A/90min"]].groupby(["Club","Year"]).sum()
    team_performances = pd.merge(team_performances,table,on=["Club","Year"])
    
    # wins_by_parameter(team_performances,param_list)
    data = salary[["Club","base_salary","Year"]].groupby(["Club","Year"]).sum()
    data2 = pd.merge(data,table,on=["Club","Year"])
    
    # print(team_performances.head())
    # print(team_performances.columns)
    # champions = team_performances[team_performances["Champion"] == True]
    # print(champions["Club"].value_counts())
    
    wins_by_parameter(team_performances,param_list,"Num_Championships")
    
    print(team_performances.columns)
    data3 = team_performances.groupby(["Club","Champion"]).size().unstack(fill_value=0).reset_index()
    # data3.fillna(0,inplace=True)
    data3['number_torunaments'] = data3[False] + data3[True]
    data3.rename(columns={True:'Num_Championships',False:'No_championship'},inplace=True)
    
    # data3['proportion_championships'] = data3[True] / data3[False]
    # data3.sort_values('proportion_championships',ascending=False,inplace=True)
    print(data3.head())
    
    
