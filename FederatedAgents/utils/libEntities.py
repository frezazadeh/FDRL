import numpy as np
from utils.libHelpers import get_configured_logger
from utils.agent_interfaces import MonitorInterface, DecisionInterface, DecisionTrackingInterface
TTI_per_sec = 1000


class BaseStation:
    def __init__(self, bs_id, monitoring_time_window, prb_chunk_size, snr_max, max_prb, min_prb, database_name, username,
                 password, venodeb, enodeb, influx_db, emulator, cell_list, log_level):
        self.ID = bs_id
        self.monitoring_time_window = monitoring_time_window
        self.prb_chunk_size = int(prb_chunk_size)
        self.max_PRBs = int(max_prb)
        self.min_PRBs = int(min_prb)
        self.allowed_prb_alloc = np.concatenate((np.array([1]), np.arange(prb_chunk_size, max_prb + 1, prb_chunk_size)))
        self.snr_max = snr_max
        self.decision_interval = None
        self.log = get_configured_logger("BS"+format(bs_id), log_level)
        self.assigned_prb = {}
        self.username = username
        self.password = password
        self.database_name = database_name
        self.venodeb = venodeb
        self.enodeb_config = enodeb
        self.influx_db = influx_db
        self.emulator = emulator
        self.monitor_interface = MonitorInterface(host=self.influx_db.split(':')[0], port=self.influx_db.split(':')[1],
                                                  username=username, password=password, database=database_name, emulator=emulator)
        self.Allocated_slices = self.venodeb['slices']
        self.Traffic_gen = self.venodeb['cells']
        enodeb_list = []
        for c in self.Traffic_gen:
            for e in self.enodeb_config:
                if c in e['cells']:
                    enodeb_list.append(e)
        self.decision_interfaces = [DecisionInterface(prb_list=self.allowed_prb_alloc, enodeb=e['ip_port'], emulator=emulator) for e in enodeb_list]
        self.decision_tracking_interface = DecisionTrackingInterface(host=self.influx_db.split(':')[0],
                                                                      port=self.influx_db.split(':')[1],
                                                                      username=username, password=password,
                                                                      database=database_name, emulator=emulator, veNB=venodeb['id'])

    def assign_slice(self, new_slice):
        pass

    def assign_prb(self, sl_id, prb):
        self.assigned_prb[sl_id] = prb

    def update_queues(self, curr_slice, decision_interval_index, decision_window):
        self.decision_interval = decision_interval_index

        traffic_request = None
        measured_channel_capacity = None
        buffer_size = None
        buffer_latency = None
        dropped_traffic = None

        return buffer_size, buffer_latency, dropped_traffic, measured_channel_capacity


