"""
Create a new Python file (e.g., lab1_ex2.py) and write a script that uses 
the psutil package to monitor the battery level of your laptop and check if 
the power is plugged. 

Every 5 seconds, print the battery status (battery level and power plugged) 
using the following format:

year-month-day hour:minute:second.microseconds - mac_address:battery = battery_level
year-month-day hour:minute:second.microseconds - mac_address:power = power_plugged

where mac_address is the MAC address of your network card, battery_level is the
battery level in percentage, and power_plugged is an integer equal to 1 if the
power is plugged, 0 otherwise.

Example:

2022-10-01 19:21:51.699254 - 0xf0b61e0bfe09:battery = 100
2022-10-01 19:21:51.699254 - 0xf0b61e0bfe09:power = 1
2022-10-01 19:21:56.701326 - 0xf0b61e0bfe09:battery = 100
2022-10-01 19:21:56.701326 - 0xf0b61e0bfe09:power = 1
"""

"""
A redis TimeSeries consists of a list of linked chunks. Each one contains a header and a set of records. A record is a key value pair (timestamp and value) 
What can we do with TimeSeries:
- compression: active by default
- retention: is 0 by default
- aggregation: can be sum, max, min, avg ...
"""
#Imports
import psutil
import uuid
import time
from datetime import datetime
import argparse as ap
import redis

#-----------------------------------------------------------------------------------------------------------------------
#Functions
def correct_timezone(x):
    day, hour = x.split(" ")
    h,m,s = hour.split(":")
    h = str( int(h) - 2 )
    result = day + " " + h + ":" + m + ":" + s
    return result

def safe_ts_create(key):
    try:
        #to create a redis time series
        redis_client.delete(key)
        redis_client.ts().create(key)   
    except redis.ResponseError:
        pass 

#-----------------------------------------------------------------------------------------------------------------------

parser = ap.ArgumentParser(description='You can choose betwenn --host --port --user --password')
parser.add_argument('--host', type=str, help='Insert the host')
parser.add_argument('--port', type=int, help='Insert the host')
parser.add_argument('--user', type=str, help='Insert the host')
parser.add_argument('--password', type=str, help='Insert the host')
args = parser.parse_args()


REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USER = args.user
REDIS_PASSWORD = args.password
mac_address = hex(uuid.getnode())


redis_client = redis.Redis(
    host=REDIS_HOST,
    port= REDIS_PORT,
    password = REDIS_PASSWORD,
    username = REDIS_USER
    )

# ping command is used to test if the connection works
print('Is connected:', redis_client.ping())

#Create the two time series: battery, power
mac_battery_name = mac_address + ':battery'
mac_power_name = mac_address + ':power'
mac_plugged_s_name = mac_address + ":plugged_seconds"
safe_ts_create(mac_battery_name)
safe_ts_create(mac_power_name)
safe_ts_create(mac_plugged_s_name) #FORSE FAR PASSARE POI LA CHUNK SIZE COME PARAMETRO A QUESTE FUNZIONI chunk_size=128


# Create the aggregation rule for the aggregation
bucket_duration_in_ms = 60 * 60 *24 * 1000 # 24h expressed in ms
redis_client.ts().createrule(mac_power_name, mac_plugged_s_name, aggregation_type='sum', bucket_size_msec=bucket_duration_in_ms)


#Create the retention rule to store only the period we are interested in 
"""EXAMPLE
# if I wanto retention only of the last 24h
one_day_in_ms = 24 * 60 * 60 * 1000     # redis always wants ms
redis_client.ts().alter('temperature', retention_msec=one_day_in_ms)
"""
upper_bound_5MB_in_ms = 3276800 * 1000
redis_client.ts().alter(mac_battery_name, retention_msec = upper_bound_5MB_in_ms)
redis_client.ts().alter(mac_power_name, retention_msec = upper_bound_5MB_in_ms)

upper_bound_1MB_in_ms = 655360 * 24 * 60 * 60 * 1000
redis_client.ts().alter(mac_plugged_s_name, retention_msec= upper_bound_1MB_in_ms)



while True:
    timestamp = time.time()
    battery_level = psutil.sensors_battery().percent
    power_plugged = int(psutil.sensors_battery().power_plugged)
    timestamp_ms = int(timestamp * 1000)
    formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

    #Printing the data for visualization
    print(f'{formatted_datetime} - {mac_address}:battery = {battery_level}')
    print(f'{formatted_datetime} - {mac_address}:power = {power_plugged}')

    #Addition of the data to the time series
    redis_client.ts().add(mac_battery_name, timestamp_ms, battery_level)
    redis_client.ts().add(mac_power_name, timestamp_ms, power_plugged)


    time.sleep(1) #every second