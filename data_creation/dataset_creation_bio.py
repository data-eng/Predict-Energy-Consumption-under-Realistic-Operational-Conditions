import pandas as pd

# ME_FO_consumption

def create_dataframe():
    data_dir = "./bio_data"

    ship_1 = pd.read_csv(f'{data_dir}/ship_1.csv')
    ship_2 = pd.read_csv(f'{data_dir}/ship_2.csv')
    ship_3 = pd.read_csv(f'{data_dir}/ship_3.csv')

    ship_1['datetime'] = pd.to_datetime(ship_1['measurement_time'])
    ship_2['datetime'] = pd.to_datetime(ship_2['measurement_time'])
    ship_3['datetime'] = pd.to_datetime(ship_3['measurement_time'])

    ship_1 = ship_1.drop(columns=['measurement_time', 'AE_FO_consumption'])
    ship_2 = ship_2.drop(columns=['measurement_time', 'AE_FO_consumption'])
    ship_3 = ship_3.drop(columns=['measurement_time', 'AE_FO_consumption'])

    ship_1 = ship_1.sort_values(by='datetime').set_index('datetime')
    ship_2 = ship_2.sort_values(by='datetime').set_index('datetime')
    ship_3 = ship_3.sort_values(by='datetime').set_index('datetime')

    ship_1 = ship_1.resample('5min').sum().reset_index()
    ship_2 = ship_2.resample('5min').sum().reset_index()
    ship_3 = ship_3.resample('5min').sum().reset_index()

    train_perc = 0.8

    ship_1_train = ship_1.iloc[:int(ship_1.shape[0] * train_perc)]
    ship_1_test = ship_1.iloc[int(ship_1.shape[0] * train_perc):]

    ship_2_train = ship_2.iloc[:int(ship_2.shape[0] * train_perc)]
    ship_2_test = ship_2.iloc[int(ship_2.shape[0] * train_perc):]

    ship_3_train = ship_3.iloc[:int(ship_3.shape[0] * train_perc)]
    ship_3_test = ship_3.iloc[int(ship_3.shape[0] * train_perc):]

    ship_1_train.to_csv(f'{data_dir}/ship_1_train.csv')
    ship_1_test.to_csv(f'{data_dir}/ship_1_test.csv')

    ship_2_train.to_csv(f'{data_dir}/ship_2_train.csv')
    ship_2_test.to_csv(f'{data_dir}/ship_2_test.csv')

    ship_3_train.to_csv(f'{data_dir}/ship_3_train.csv')
    ship_3_test.to_csv(f'{data_dir}/ship_3_test.csv')


create_dataframe()