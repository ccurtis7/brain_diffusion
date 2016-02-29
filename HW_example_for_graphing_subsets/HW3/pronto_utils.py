import wget
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
'''import seaborn; seaborn.set()'''

def download_if_needed(URL, filename):
    """
    Downloads from URL to filename unless filename already exists
    """
    if os.path.exists(filename):
        print filename,'already exists'
        return
    else:
        print 'Downloading',filename,'...OK'
        wget.download(URL)

def get_pronto_data():
    """
    Download pronto data, unless already downloaded
    """
    download_if_needed('https://s3.amazonaws.com/pronto-data/open_data_year_one.zip','open_data_year_one.zip')

def get_station_data():
    """
    Grabs individual .csv from within .zip file
    """
    get_pronto_data()
    zf = zipfile.ZipFile('open_data_year_one.zip')
    file_handle = zf.open('2015_station_data.csv')
    return pd.read_csv(file_handle, parse_dates='online')

def get_status_data():
    """
    Grabs individual .csv from within .zip file
    """
    get_pronto_data()
    zf = zipfile.ZipFile('open_data_year_one.zip')
    file_handle = zf.open('2015_status_data.csv')
    d = pd.read_csv(file_handle, parse_dates='time')
    da = pd.DatetimeIndex(d['time'])
    return da.groupby([da.date,'station_id'])['bikes_available'].mean()

def get_trip_data():
    """
    Grabs individual .csv from within .zip file
    """
    get_pronto_data()
    zf = zipfile.ZipFile('open_data_year_one.zip')
    file_handle = zf.open('2015_trip_data.csv')
    return pd.read_csv(file_handle, parse_dates=['starttime','stoptime'])

def get_weather_data():
    """
    Grabs weather .csv from within .zip file
    """
    get_pronto_data()
    zf = zipfile.ZipFile('open_data_year_one.zip')
    file_handle = zf.open('2015_weather_data.csv')
    return pd.read_csv(file_handle, parse_dates='Date')

def get_trips_and_weather():
    """
    Combines the trip data and the weather data
    """
    data = get_trip_data()
    wdata = get_weather_data()
    date = pd.DatetimeIndex(data['starttime'])
    '''
    Pivot table = two-dimensional groupby
    '''
    trips_by_date = data.pivot_table('trip_id', aggfunc='count', index=date.date, columns='usertype')
    wdata = wdata.set_index('Date')
    wdata.index = pd.DatetimeIndex(wdata.index)
    wdata = wdata.iloc[:-1]
    return wdata.join(trips_by_date)

def get_tripdurations_and_weather():
    trips = get_trip_data()
    wdata = get_weather_data()
    '''Prepare the weather data'''
    wdata = wdata.set_index('Date')
    wdata.index = pd.DatetimeIndex(wdata.index)
    wdata = wdata.iloc[:-1]
    '''Prepare the trip data and join a daily mean trip duration to the weather data'''
    tdate = pd.DatetimeIndex(trips['starttime'])
    jdata = wdata.join(trips.groupby(tdate.date)['tripduration'].mean())
    return jdata

def plot_weekly_data(DayOfWeek, Rain, Xaxis):
    if Rain==True:
        rdata = jdata[jdata.Events == 'Rain']
        rddata = rdata[pd.DatetimeIndex(rdata.index).dayofweek==DayOfWeek]
        rddata.plot.scatter(Xaxis,'tripduration')
    else:
        nrdata = jdata[jdata.Events != 'Rain']
        nrddata = nrdata[pd.DatetimeIndex(nrdata.index).dayofweek==DayOfWeek]
        nrddata.plot.scatter(Xaxis,'tripduration')

def get_demographic_data():
    trips = get_trip_data()
    wdata = get_weather_data()
    '''Prepare the weather data'''
    wdata = wdata.set_index('Date')
    wdata.index = pd.DatetimeIndex(wdata.index)
    wdata = wdata.iloc[:-1]
    wdata_trunc = wdata['Precipitation_In ']
    wdata_trunc2 = wdata['Min_Visibility_Miles ']
    wdata_trunc3 = wdata['Max_Wind_Speed_MPH ']
    '''Prepare the trip data'''
    trips_before1984 = trips[trips.birthyear < 1984]
    trips_after1984 = trips[trips.birthyear >= 1984]
    trips_before1984['Before 1984'] = trips_before1984['birthyear']
    trips_after1984['During or after 1984'] = trips_after1984['birthyear']
    mtrips_before1984 = trips_before1984[trips_before1984.gender == 'Male']
    mtrips_before1984['Male Before 1984'] = mtrips_before1984['birthyear']
    ftrips_before1984 = trips_before1984[trips_before1984.gender == 'Female']
    ftrips_before1984['Female Before 1984'] = ftrips_before1984['birthyear']
    otrips_before1984 = trips_before1984[trips_before1984.gender == 'Other']
    otrips_before1984['Other Before 1984'] = otrips_before1984['birthyear']
    mtrips_after1984 = trips_after1984[trips_after1984.gender == 'Male']
    mtrips_after1984['Male During or after 1984'] = mtrips_after1984['birthyear']
    ftrips_after1984 = trips_after1984[trips_after1984.gender == 'Female']
    ftrips_after1984['Female During or after 1984'] = ftrips_after1984['birthyear']
    otrips_after1984 = trips_after1984[trips_after1984.gender == 'Other']
    otrips_after1984['Other During or after 1984'] = otrips_after1984['birthyear']
    '''Creating a demographics file from the trip file'''
    date = pd.DatetimeIndex(trips['starttime'])
    trips_by_date_type = trips.pivot_table('trip_id', aggfunc='count', index=date.date, columns=['usertype'])
    trips_by_date_gender = trips.pivot_table('trip_id', aggfunc='count', index=date.date, columns=['gender'])
    trips_by_date_ageb1984 = trips_before1984.groupby(pd.DatetimeIndex(trips_before1984['starttime']).date)['Before 1984'].count()
    mtrips_by_date_ageb1984 = mtrips_before1984.groupby(pd.DatetimeIndex(mtrips_before1984['starttime']).date)['Male Before 1984'].count()
    ftrips_by_date_ageb1984 = ftrips_before1984.groupby(pd.DatetimeIndex(ftrips_before1984['starttime']).date)['Female Before 1984'].count()
    otrips_by_date_ageb1984 = otrips_before1984.groupby(pd.DatetimeIndex(otrips_before1984['starttime']).date)['Other Before 1984'].count()
    trips_by_date_agea1984 = trips_after1984.groupby(pd.DatetimeIndex(trips_after1984['starttime']).date)['During or after 1984'].count()
    mtrips_by_date_agea1984 = mtrips_after1984.groupby(pd.DatetimeIndex(mtrips_after1984['starttime']).date)['Male During or after 1984'].count()
    ftrips_by_date_agea1984 = ftrips_after1984.groupby(pd.DatetimeIndex(ftrips_after1984['starttime']).date)['Female During or after 1984'].count()
    otrips_by_date_agea1984 = otrips_after1984.groupby(pd.DatetimeIndex(otrips_after1984['starttime']).date)['Other During or after 1984'].count()
    demos = trips_by_date_type.join([trips_by_date_gender,trips_by_date_ageb1984,trips_by_date_agea1984,mtrips_by_date_ageb1984,ftrips_by_date_ageb1984,otrips_by_date_ageb1984,mtrips_by_date_agea1984,ftrips_by_date_agea1984,otrips_by_date_agea1984])
    '''Joining the demographics and weather files'''
    demos2 = demos.join(wdata_trunc)
    demos3 = demos2.join(wdata_trunc2)
    jdemos = demos3.join(wdata_trunc3)
    jdemos['All Annual Members'] = jdemos['Annual Member']
    return jdemos

def plot_demographics(IncludeShortTermPassHolders, Gender, Birthyear):
    jdemos = get_demographic_data()
    fig, ax = plt.subplots(3, figsize=(16,8), sharex=True)
    if IncludeShortTermPassHolders==True:
        jdemos['Short-Term Pass Holder'].plot(ax=ax[0], title='Number of rides per day by membership type', legend=True)
    if Birthyear=='All':
        jdemos[Gender].plot(ax=ax[0], title='Number of rides per day by membership type', legend=True)
    if Birthyear!='All':
        if Gender!='All Annual Members':
            jdemos[Gender+' '+Birthyear].plot(ax=ax[0], title='Number of rides per day by membership type', legend=True)
        else:
            jdemos[Birthyear].plot(ax=ax[0], title='Number of rides per day by membership type', legend=True)
    jdemos['Precipitation_In '].plot(ax=ax[1], title='Precipitation')
    jdemos['Min_Visibility_Miles '].plot(ax=ax[2], title='Wind and visibility', legend=True)
    jdemos['Max_Wind_Speed_MPH '].plot(ax=ax[2], title='Wind and visibility', legend=True)

def plot_daily_totals():
    """
    Creates a plot for comparison of annual memberships and short-term passes vs date
    """
    data = get_trips_and_weather()
    fig, ax = plt.subplots(2, figsize=(14,6),sharex=True)
    data['Annual Member'].plot(ax=ax[0], title='Annual Member')
    data['Short-Term Pass Holder'].plot(ax=ax[1], title='Short-term Pass Holder')
    fig.savefig('daily_totals.png')
