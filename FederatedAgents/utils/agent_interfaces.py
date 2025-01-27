#!/usr/bin/env python3
# import json
# import subprocess
# import requests
# # Disable SSL certificate validation
# from requests.packages.urllib3.exceptions import InsecureRequestWarning
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from websocket import create_connection
import json
from influxdb import InfluxDBClient
from datetime import datetime
import argparse
import configparser
import time
import zmq
import sys

INFLUX_DATABASE = 'SRSLTE'
API_URL = 'http://localhost:3000'
API_USERNAME = 'admin'
API_PASSWORD = 'admin'
EMULATOR = 'srslte'

E_NODE_B = '10.10.244.69:5000' #'tcp://10.10.244.69:5000'
PRB_LIST = [1, 10, 20, 30, 40, 50]


class GlobalInfluxInteraction:
    def __init__(self, host='localhost', port=8086, username=API_USERNAME, password=API_PASSWORD,
                 database=INFLUX_DATABASE, emulator=EMULATOR):
        self.emulator = emulator
        self.client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)
        self.databases = [i['name'] for i in self.client.get_list_database()]
        self.measurements = [i['name'] for i in self.client.get_list_measurements()]
        self.fields = self.client.query('SHOW FIELD KEYS').raw['series']

    def clean_influx(self):
        try:
            for meas in self.measurements:
                self.client.drop_measurement(meas)
            print("InfluxDB cleaned")
        except:
            print("InfluxDB Exception occurred")
        return 0


class WsConnection:
    def __init__(self, ip='localhost', port='9001'):
        self.ip = ip
        self.port = port
        self.ws = create_connection("ws://{}:{}".format(self.ip, self.port))
        print(json.loads(self.ws.recv()))
        self.messages = {'config_set': '{\"message\":\"config_set\", \"cells\":{\"1\" :{\"pdsch_fixed_rb_alloc\":true,\"pdsch_fixed_rb_start\":1, \"pdsch_fixed_l_crb\":270} }}}'}

    def query(self, message):
        try:
            self.ws.send(message)
            return json.loads(self.ws.recv())
        except Exception as e:
            print("exception:\n")
            print(e)
            print("no response...")
            return None

    def get_messages_list(self):
        for m in self.messages:
            print(m, self.messages[m])


class MonitorInterface:
    def __init__(self, host='localhost', port=8086, username=API_USERNAME, password=API_PASSWORD,
                 database=INFLUX_DATABASE, emulator=EMULATOR):
        self.emulator = emulator
        self.client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)
        self.databases = [i['name'] for i in self.client.get_list_database()]
        self.measurements = [i['name'] for i in self.client.get_list_measurements()]
        self.fields = self.client.query('SHOW FIELD KEYS').raw['series']

    def get_measurements_names(self):
        return self.fields[0], self.fields[1]

    def get_measurements(self, measurement_name=None, metric=None, time_window=None, operation='mean', filter_out=None):
        if self.emulator == 'amarisoft':
            if isinstance(time_window, int):
                query = 'SELECT ' + operation + '("'+metric+'") FROM "'+measurement_name
                query = query + '" WHERE '
                if filter_out is not None:
                    query = query + filter_out + ' AND '
                query = query + 'time >= now() -' + str(time_window) + 's fill(null)'
                try:
                    measurement = self.client.query(query).raw['series'][0]['values'][0][1]
                except:
                    measurement = 0.0
                    print('NO WORK!')
            return measurement

        if self.emulator == 'srslte':
            if isinstance(time_window, int):
                query = 'SELECT ' + operation + '("{}") FROM "{}" WHERE ("cell" = \'0x19B\') AND time >= now() - ' + \
                        str(time_window) + 's fill(null)'
            elif operation == 'last':
                query = 'SELECT last("{}") FROM "{}" WHERE ("cell" = \'0x19B\') fill(null)'

            pos = self.find_index_in_raw(measurement_name)
            measurements = {}
            for index, value in enumerate(self.fields[pos]['values']):
                try:
                    measurements[value[0]] = self.client.query(query.format(value[0], self.fields[pos]['name'])).raw['series'][0]['values'][0][1]
                except:
                    print('{} not available for {}'.format(value[0], self.fields[pos]['name']))
            return measurements

    def find_index_in_raw(self, name):
        for index in range(len(self.fields)):
            if self.fields[index]['name'] == name:
                return index
        return None


class DecisionTrackingInterface:
    def __init__(self, host='localhost', port=8086, username=API_USERNAME, password=API_PASSWORD,
                 database=INFLUX_DATABASE, emulator=EMULATOR, veNB='1'):
        self.emulator = emulator
        self.client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)
        self.databases = [i['name'] for i in self.client.get_list_database()]
        self.measurements = [i['name'] for i in self.client.get_list_measurements()]
        self.fields = self.client.query('SHOW FIELD KEYS').raw['series']
        self.veNB = veNB

    def write_decision(self, dl_prb_allocation, ul_prb_allocation, dl_allocation_gap, ul_allocation_gap):
        json_payload = []
        for slice, prb in enumerate(dl_prb_allocation):
            ticker = 'slice_' + str(slice)
            data = {
                "measurement": "prb_allocation",
                "tags": {"ticker": ticker, "veNB": self.veNB},
                "fields": {'dl_prb': float(prb), 'ul_prb': float(ul_prb_allocation[slice])}
            }
            data['fields']['size'] = sys.getsizeof(str(data))

            json_payload.append(data)
            data_2 = {
                "measurement": "prb_allocation",
                "tags": {"ticker": ticker, "veNB": self.veNB},
                "fields": {'dl_gap': float(dl_allocation_gap[slice]), 'ul_gap': float(ul_allocation_gap[slice])}
            }
            data_2['fields']['size'] = sys.getsizeof(str(data_2))

            json_payload.append(data_2)
        return self.client.write_points(json_payload)

    def write_reward(self, reward):
        json_payload = []
        for slice, value in enumerate(reward):
            ticker = 'slice_' + str(slice)
            data = {
                "measurement": "reward",
                "tags": {"ticker": ticker, "veNB": self.veNB},
                "fields": {'r': float(value)}
            }
            data['fields']['size'] = sys.getsizeof(str(data))
            json_payload.append(data)
        return self.client.write_points(json_payload)

class DecisionInterface:
    def __init__(self, prb_list=PRB_LIST, enodeb=E_NODE_B, emulator=EMULATOR):
        self.emulator = emulator
        if emulator == 'amarisoft':
            self.socket = WsConnection(enodeb.split(':')[0], enodeb.split(':')[1])
        if emulator == 'srslte':
            self.context = zmq.Context(1)
            # print("Connecting to hello world server")
            self.socket = self.context.socket(zmq.REQ)  # PUSH
            # socket.bind("tcp://*:5556")  # Connect to external port of docker

            # socket.connect("tcp://localhost:5556") # Connect to external port of container
            self.socket.connect("tcp://" + enodeb)  # Connect to external port of container
            self.prb_list = prb_list

    def request(self, decision_dl, decision_ul, cell_id=1):
        if self.emulator == 'amarisoft':
            self.request_amarisoft(decision_dl, decision_ul, cell_id)
        if self.emulator == 'srslte':
            self.request_srslte(decision_dl, decision_ul, cell_id)

    def request_amarisoft(self, decision_dl, decision_ul, cell_id=1):
        message = '{\"message\":\"config_set\", \"cells\":{\"'+str(cell_id)+'\" :{\"pdsch_fixed_rb_alloc\":true,\"pdsch_fixed_rb_start\":1, \"pdsch_fixed_l_crb\":' + str(decision_dl) +'} }}'
        reply = self.socket.query(message)  # todo: there is an error here with the PRB allocation for the second agent
        print(reply)
        #time.sleep(1)

    def request_srslte(self, decision_dl, decision_ul):
        ctrl_message = format(decision_dl) + ',' + format(decision_ul)  # downlink, uplink

        print("Sending request " + format(ctrl_message))
        self.socket.send_string(ctrl_message, zmq.NOBLOCK)
        time.sleep(5)

        # Get the reply.
        ack_message = self.socket.recv()
        print("Received reply PRB_DL {}, PRB_UL {} [ {} ]".format(decision_dl, decision_ul, ack_message))
        time.sleep(1)
