# FROM https://www.thepythoncode.com/article/make-a-network-usage-monitor-in-python
import psutil
import time
import os
import pandas as pd
from influxdb import InfluxDBClient
import argparse
import configparser
from datetime import datetime

#### Usage:
##### python3 traffic_monitoring <ID> <INTERFACE> <UPDATE_TIME>

def get_size(bytes):
    """
    Returns size of bytes in a nice format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024

def bytes_to_bits(bytes):
    return bytes*8

def main():

    # ID = 1
    # INTERFACE = 'eth0'
    # UPDATE_DELAY = 1  # in seconds

    # ID = str(argv[1])
    # INTERFACE = str(argv[2])
    # UPDATE_DELAY = int(argv[3])

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='1')
    parser.add_argument('--iface', type=str, default='eth0')
    parser.add_argument('--time', type=float, default='0.5')
    parser.add_argument('--config', type=str, default='traffic_generator.config')
    args = parser.parse_args()
    config_name = args.config.replace('/', ',').replace('.', ',').split(',')[1]

    ID = args.id
    INTERFACE = args.iface
    UPDATE_DELAY = args.time

    # Read configuration file
    config = configparser.RawConfigParser(defaults=None, strict=False)
    config.read(args.config)

    #### Send data to influxDB
    client = InfluxDBClient(config.get('INFLUX', 'ip'),
                            config.get('INFLUX', 'port'),
                            config.get('INFLUX', 'user'),
                            config.get('INFLUX', 'password'),
                            config.get('INFLUX', 'db_name'))

    if {'name': config.get('INFLUX', 'db_name')} not in client.get_list_database():
        client.create_database(config.get('INFLUX', 'db_name'))

    # get the network I/O stats from psutil on each network interface
    # by setting `pernic` to `True`
    io = psutil.net_io_counters(pernic=True)

    while True:

        # sleep for `UPDATE_DELAY` seconds
        time.sleep(UPDATE_DELAY)
        # now = datetime.now()

        # get the network I/O stats again per interface
        io_2 = psutil.net_io_counters(pernic=True)
        # initialize the data to gather (a list of dicts)
        data = []
        for iface, iface_io in io.items():
            # new - old stats gets us the speed
            download_speed, upload_speed = io_2[iface].bytes_sent - iface_io.bytes_sent, io_2[iface].bytes_recv - iface_io.bytes_recv
            data.append({
                "iface": iface,
                "Upload": bytes_to_bits(io_2[iface].bytes_recv),
                "Download": bytes_to_bits(io_2[iface].bytes_sent),
                "Download Speed": bytes_to_bits(download_speed / UPDATE_DELAY),
                "Upload Speed": bytes_to_bits(upload_speed / UPDATE_DELAY),
            })
        # update the I/O stats for the next iteration
        io = io_2
        # construct a Pandas DataFrame to print stats in a cool tabular style
        df = pd.DataFrame(data)
        # sort values per column, feel free to change the column
        df.sort_values("Download", inplace=True, ascending=False)
        # clear the screen based on your OS
        os.system("cls") if "nt" in os.name else os.system("clear")
        # print the stats
        # print(df.to_string())

        if INTERFACE not in list(df['iface']):
            print('ERROR: Interface not available, switching to localhost..')
            INTERFACE = 'lo'

        json_body = [
            {
                "measurement": "traffic",
                "tags": {
                    "traffic_gen_ID": int(ID)
                },
                #"time": now,
                "fields": {
                    "Download": int(df[df['iface'] == INTERFACE]['Download'].values),
                    "Upload": int(df[df['iface'] == INTERFACE]['Upload'].values),
                    "Download Speed": int(df[df['iface'] == INTERFACE]['Download Speed'].values),
                    "Upload Speed": int(df[df['iface'] == INTERFACE]['Upload Speed'].values)
                }
            }
        ]

        client.write_points(json_body)

if __name__ == "__main__":
   main()
