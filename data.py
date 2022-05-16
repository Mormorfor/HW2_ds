import pandas as pd
from datetime import datetime

def load_data(path):
    """
    Loads csv file into a pandas data frame
    :param path: path to the csv file on the machine
    :return: data frame built from the csv file
    """
    df = pd.read_csv(path)
    return df

def add_new_columns(df):
    """
    Adds new columns to a given data frame.
    Columns are: season name, hour, day, month, year, is weekend holiday
    :param df: data frame of data points
    :return:  edited data frame (with additional columns)
    """
    df['season_name'] = df['season'].apply(num_to_string_season)
    df['Hour'] = df.apply(lambda x : get_time_element(x.timestamp, 'hour'), axis = 1)
    df['Day'] = df.apply(lambda x: get_time_element(x.timestamp, 'day'), axis=1)
    df['Month'] = df.apply(lambda x : get_time_element(x.timestamp, 'month'), axis = 1)
    df['Year'] = df.apply(lambda x : get_time_element(x.timestamp, 'year'), axis = 1)
    df['is_weekend_holiday'] = df.apply(lambda  x: set_holiday_weekend(x['is_weekend'], x['is_holiday']), axis = 1);
    df['t_diff'] = df.apply(lambda  x: temp_difference_calc(x['t1'], x['t2']), axis = 1);

    return df

def data_analysis(df):
    """
    Analyzes data, prints correlation between every value set, gives five most and five least correlated features
    :param df: data frame of data points
    :return: void
    """
    print("describe output:")
    print(df.describe().to_string())
    print()

    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()


    corr_dct = create_corr_dict(df)
    sorted_dict = sort_corr_dict(corr_dct)

    keys_list = list(sorted_dict)
    print("Highest correlated are: ")
    length = len(sorted_dict)
    for i in range(5):
        form_val = "{:.6f}".format(sorted_dict[keys_list[length - i - 1]])
        print(f"{i+1}. {keys_list[length - i - 1]} with {form_val}")
    print()

    print("Lowest correlated are: ")
    for i in range(5):
        form_val = "{:.6f}".format(sorted_dict[keys_list[i]])
        print(f"{i+1}. {keys_list[i]} with {form_val}")


    print()
    means_by_season = df.groupby('season_name').mean()
    means_by_season_dic = means_by_season.to_dict()

    for season in means_by_season_dic['t_diff']:
        form_seas = "{:.2f}".format(means_by_season_dic['t_diff'][season])
        print(f"{season} average t_diff is {form_seas}")
    form_all = "{:.2f}".format(df['t_diff'].mean())
    print(f"All average t_diff is {form_all}")


def create_corr_dict(df):
    """
    Helping function - creates a dictionary of correlated pairs of values
    :param df:  data frame of data points
    :return: dictionary with key - touple of two column names and value - (absolute) correlation between them
    """
    dct = {}
    for col1 in range(0, len(df.columns) - 1):
        for col2 in range(col1 + 1, len(df.columns)):
             if (df.iloc[:, col1].dtype != object) & (df.iloc[:, col2].dtype != object):
                dct[(df.columns[col1], df.columns[col2])] = abs(df.iloc[:, col1].corr(df.iloc[:, col2]))
    return dct

def sort_corr_dict(corr_dct):
    """
    Helping function - sorts a dictionary from low values to high
    :param corr_dct: dictionary of correlations between features
    :return:  sorted dictionary
    """
    sorted_values = sorted(corr_dct.values())
    sorted_dict = {}
    for i in sorted_values:
        for k in corr_dct.keys():
            if corr_dct[k] == i:
                sorted_dict[k] = corr_dct[k]
    return sorted_dict

def temp_difference_calc(t1, t2):
    """
    Helping function - calculate a difference between two numbers
    :param t1: number 1
    :param t2: number 2
    :return: difference between t1 and t2
    """
    return t2 - t1

def set_holiday_weekend(weekend, holiday):
    """
    Helping function - returns value according to values of weekend and holiday
    :param weekend: 0/1 value of is_weekend
    :param holiday: 0/1 value of is_holiday
    :return:0 - weekend=0 and holiday=0
            1 - weekend=1 and holiday=0
            2 - weekend=0 and holiday=1
            3 - weekend=1 and holiday=1
    """
    if (weekend == 0) & (holiday == 0):
        return 0
    if (weekend == 1) & (holiday == 0):
        return 1
    if (weekend == 0) & (holiday == 1):
        return 2
    if (weekend == 1) & (holiday == 1):
        return 3

def get_time_element(timestamp, param):
    """
     Helping function - cuts a specific element out of complete time stamp
    :param timestamp: a d/m/Y H:M formatted time stamp
    :param param: string - which element of the time stamp is needed
    :return: a chosen by parameter element of the time stamp
    """
    time = datetime.strptime(timestamp, "%d/%m/%Y %H:%M")
    if param == 'hour':
        return time.hour
    elif param == 'year':
        return time.year
    elif param == 'month':
        return time.month
    elif param == 'day':
        return time.day

def num_to_string_season(num):
    """
     Helping function - transforms numerical values into string - season names
    :param num: a number from 0 to 3
    :return: 0 - spring
             1 - summer
             2 - fall
             3 - winter
    """
    if num == 0:
        return "spring"
    elif num == 1:
        return "summer"
    elif num == 2:
        return "fall"
    elif num == 3:
        return "winter"

