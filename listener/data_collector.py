from websocket import create_connection
import json
from influxdb import InfluxDBClient
from datetime import datetime
import argparse
import configparser
import time
import sys


class WsConnection:
    def __init__(self, ip='localhost', port='9001'):
        self.ip = ip
        self.port = port
        self.ws = create_connection("ws://{}:{}".format(self.ip, self.port))
        print(json.loads(self.ws.recv()))
        self.messages = {'ue_get': "{\"message\":\"ue_get\" ,\"stats\"=true}",
                         'bs_get': "{\"message\":\"stats\" ,\"stats\"=true}"}

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

    def ue_get(self, print_flag=False):
        try:
            reply = self.query(self.messages['ue_get'])
            reply.pop('message')
            for ue_ind, ue in enumerate(reply['ue_list']):
                for cell_ind, cell in enumerate(ue['cells']):
                    for key in cell:
                        reply['ue_list'][ue_ind][key] = float(reply['ue_list'][ue_ind]['cells'][cell_ind][key])
                reply['ue_list'][ue_ind].pop('cells')
                for qos_flow_ind, qos_flow in enumerate(ue['qos_flow_list']):
                    for key in qos_flow:
                        reply['ue_list'][ue_ind][key] = float(reply['ue_list'][ue_ind]['qos_flow_list'][qos_flow_ind][key])
                reply['ue_list'][ue_ind].pop('qos_flow_list')
            reply = reply['ue_list']
            return reply
        except Exception as v:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(v).__name__, v.args)
            print(message)
            return None

    def bs_get(self, print_flag=False):
        reply = self.query(self.messages['bs_get'])
        # adapt the reply to influx
        if reply is None:
            print("No reply received")
            return None, None, None
        reply.pop('counters')
        for key in reply['cells'].keys():
            reply['cells'][key].pop('counters')
        try:
            reply['cpu'] = float(reply['cpu']['global'])
        except Exception as v:
            print(v)
            reply['cpu'] = float(0)
        try:
            rf_port_list = []
            for key_ind, key in enumerate(reply['rf_ports']):
                rf_port_list.append({'rf_port_id': key})
                for key_2 in reply['rf_ports'][key]['rxtx_delay']:
                    rf_port_list[key_ind]['rxtx_delay_' + key_2] = float(reply['rf_ports'][key]['rxtx_delay'][key_2])
            reply.pop('rf_ports')

            cell_list = []
            for key_ind, key in enumerate(reply['cells']):
                cell_list.append({'cell_id': int(key)})
                for key_2 in reply['cells'][key]:
                    cell_list[key_ind]['cell_' + key_2] = float(reply['cells'][key][key_2])
            reply.pop('cells')
            reply.pop('utc')
            reply.pop('time')
            reply.pop('message')
            reply.pop('instance_id')
            reply['duration'] = float(reply['duration'])
            return reply, rf_port_list, cell_list
        except Exception as v:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(v).__name__, v.args)
            print(message)
            return None, None, None


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='data_collector.config')
args = parser.parse_args()
config_name = args.config.replace('/', ',').replace('.', ',').split(',')[1]

# Read configuration file
config = configparser.RawConfigParser(defaults=None, strict=False)
config.read(args.config)


# Connect with BS
bs = WsConnection(config.get('BS', 'ip'), config.get('BS', 'port'))
bs_id = config.get('BS', 'ip')+':'+config.get('BS', 'port')
bs.get_messages_list()

# Interaction with influx
client = InfluxDBClient(config.get('INFLUX', 'ip'),
                        config.get('INFLUX', 'port'),
                        config.get('INFLUX', 'user'),
                        config.get('INFLUX', 'password'),
                        config.get('INFLUX', 'db_name'))

if {'name': config.get('INFLUX', 'db_name')} not in client.get_list_database():
    client.create_database(config.get('INFLUX', 'db_name'))


# monitoring
monitoring_time_window = int(config.get('BS', 'monitoring_time_window'))
while True:
    now = datetime.now()
    json_payload = []
    general, rf_ports, cells = bs.bs_get()

    if general is not None:
        data = {
            "measurement": "general",
            "tags": {"ticker": bs_id},
            #"time": now,
            "fields": general
        }
        data['fields']['size'] = sys.getsizeof(str(data))
        json_payload.append(data)

        for cell in cells:
            ticker = 'cell_id_' + str(cell['cell_id'])
            data = {
                "measurement": "cell",
                "tags": {"ticker": ticker, "BS_id": bs_id},
                #"time": now,
                "fields": cell
                }
            data['fields']['size'] = sys.getsizeof(str(data))
            json_payload.append(data)

        for rf_port in rf_ports:
            ticker = 'rf_port_id_' + str(rf_port['rf_port_id'] + '_' + bs_id)
            data = {
                "measurement": "rf_port",
                "tags": {"ticker": ticker, "BS_id": bs_id},
                #"time": now,
                "fields": rf_port
            }
            data['fields']['size'] = sys.getsizeof(str(data))
            json_payload.append(data)

        for ue in bs.ue_get():
            ticker = 'ue_' + str(ue['ran_ue_id'])
            data = {
                "measurement": "ues",
                "tags": {"ticker": ticker, "BS_id": bs_id},
                #"time": now,
                "fields": ue
            }
            data['fields']['size'] = sys.getsizeof(str(data))
            json_payload.append(data)

        # send payload
        client.write_points(json_payload)

    delta_t = datetime.now() - now
    delta_t = delta_t.seconds + delta_t.microseconds/1000000
    time.sleep(max(0.0, monitoring_time_window - delta_t))


# result
# result = client.query('select * from enb;')


