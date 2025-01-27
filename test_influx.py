from influxdb import InfluxDBClient
from datetime import datetime


# setup database
client = InfluxDBClient('localhost', 8086, 'admin', 'admin', 'mydb')
client.create_database('mydb')
client.get_list_database()

# setup payload
json_payload = []
data = {
    "measurement": "stocks",
    "tags": {"ticker": "TESLA"},
    "time": datetime.now(),
    "fields": {
        "open": 500, "close": 510
    }
}

json_payload.append(data)

# send payload
client.write_points(json_payload)

# result
result = client.query('select * from stocks;')

