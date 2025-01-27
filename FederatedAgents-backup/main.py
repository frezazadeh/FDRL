import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import configparser
import argparse
import numpy as np
import gym
from custom_BS import registration, MultiDict
from utils.libHelpers import get_configured_logger
import Agent
from MultiAgent import Multi_Agent
from utils.libEntities import BaseStation
from time import sleep
from utils.agent_interfaces import GlobalInfluxInteraction
import subprocess
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam



'''
def inter_slice_federation(model_slice_1, model_slice_2):
    fed_model = [np.zeros(model_slice_1.get_weights()[w].shape)
                 for w in range(len(model_slice_1.get_weights()))]

    curr_model1 = model_slice_1.get_weights()
    curr_model2 = model_slice_2.get_weights()
    for w in range(len(curr_model1)):
        fed_model[w] = curr_model1[w] + curr_model2[w]
    for w in range(len(fed_model)):
        fed_model[w] = fed_model[w] / 2
    return (fed_model)

'''

def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models]     # averaging outputs
    yAvg=layers.average(yModels)     # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')  
    return modelEns
    
def inter_slice_federation(model_slice_1, model_slice_2):
    models=[]
    model_slice_1.name="s1111111"
    model_slice_2.name="s2222222"
    models.append(model_slice_1)
    models.append(model_slice_2)
    model_input = Input(shape=models[0].input_shape[1:])
    modelEns = ensembleModels(models, model_input)
    return (modelEns)

def save_to_file(args):
    try:
        filename = args[0]
        object = args[1]
        np.savez(filename, object, allow_pickle=False)
        return 0
    except:
        return -1

def compute_agent_data_size(multi_agents, drl_class_name):
    weight = [0 for _ in range(len(multi_agents))]
    for ma_ind, ma in enumerate(multi_agents):
        for a in ma.agents:
            if drl_class_name == 'DDQN':
                for layer in a.train_network.get_weights():
                    weight[ma_ind] = weight[ma_ind] + 2*layer.size*layer[0].itemsize  #todo: fix data measure unit (now it's Bytes)
    return np.array(weight)

def run_commands_remote(hostname, user, command):
    ssh = subprocess.Popen(["ssh", user + "@" + hostname, command],
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    result = ssh.stdout.readlines()
    return result

def get_list_slices(hostname, user):
    list_slices = run_commands_remote(hostname, user, 'docker container ls | grep traffic_generator')
    list_slices = [str(line).split(' ')[-1].replace("\\n'",'') for line in list_slices]
    return list_slices


def main():

    # Get Logger
    log_level = "INFO"
    log = get_configured_logger(__name__, log_level)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config_BS/config.properties')
    parser.add_argument('--vBS', type=str, default='None')
    args = parser.parse_args()
    config_name = args.config.replace('/', ',').replace('.', ',').split(',')[1]
    vbs_to_instantiate = args.vBS

    # Read configuration file
    config = configparser.RawConfigParser(defaults=None, dict_type=MultiDict, strict=False)
    config.read(args.config)
    monitoring_time_window = int(config.get('Simulation', 'monitoring_time_window'))  # Decision window size in seconds (Monitoring + Decision period)
    n_decisions_per_episode = int(config.get('DRL', 'Max_Episode'))
    drl_class = getattr(Agent, config.get('DRL', '_class_ML'))
    saving_period = int(config.get('DRL', 'save_to_file_every'))
    clear_models = eval(config.get('Simulation', 'clear_models').capitalize())

    registration(n_decisions_per_episode, log_level)

    # clear db before starting
    db_interactor = GlobalInfluxInteraction(host=config.get('INFLUX', 'INFLUX_DB').split(':')[0],
                                            port=config.get('INFLUX', 'INFLUX_DB').split(':')[1],
                                            username=config.get('INFLUX', 'API_USERNAME'),
                                            password=config.get('INFLUX', 'API_PASSWORD'),
                                            database=config.get('INFLUX', 'DATABASE_NAME'),
                                            emulator=config.get('BS_GENERIC', 'emulator'))

    #db_interactor.clean_influx()

    sleep(2) # wait for setting up

    # getting the slices list under the assumption of one slice per traffic generator
    traffic_gen_list = get_list_slices(config.get('BRIDGE_VM', 'IP'), config.get('BRIDGE_VM', 'USER'))
    nr_cells = len(traffic_gen_list)

    vbs_list = [bs for bs in config.keys() if 'VBS' in bs]
    if vbs_to_instantiate != 'none':
        other_vbs = [bs for bs in config.keys() if 'VBS' in bs]
        if vbs_to_instantiate in vbs_list:
            vbs_list = [vbs_to_instantiate]
            other_vbs.pop(other_vbs.index(vbs_to_instantiate))
        else:
            raise IOError(vbs_to_instantiate + ' is not in ' +  config_name)
    else:
        other_vbs = []


    veNB_config = [{'id': bs, 'cells': config.get(bs, 'CELLS').replace(' ','').split(','),
                    'slices': config.get(bs, 'SLICES').replace(' ','').split(','), 'federation':[]}
                   for bs in vbs_list]

    for bs in veNB_config:
        for slice_id in bs['slices']:
            bs['federation'].append(eval(config.get('SLICE'+str(slice_id), 'FEDERATION').capitalize()))



    eNB_config = [{'id': bs, 'ip_port': config.get(bs, 'E_NODE_B'),
                    'cells': config.get(bs, 'CELLS').replace(' ','').split(',')}
                   for bs in config.keys() if 'BS' in bs and 'GENERIC' not in bs and 'V' not in bs]

    # Connect to the basestation
    base_stations = [BaseStation(
        veNB['id'],
        monitoring_time_window,
        int(config.get('BS_GENERIC', 'chunk_size')),
        int(config.get('BS_GENERIC', 'SNR_MAX')),
        int(config.get('BS_GENERIC', 'MAX_PRBs')),
        int(config.get('BS_GENERIC', 'MIN_PRBs')),
        config.get('INFLUX', 'DATABASE_NAME'),
        config.get('INFLUX', 'API_USERNAME'),
        config.get('INFLUX', 'API_PASSWORD'),
        veNB,
        eNB_config,
        config.get('INFLUX', 'INFLUX_DB'),
        config.get('BS_GENERIC', 'emulator'),
        traffic_gen_list,
        log_level)
        for veNB in veNB_config]

    log.info("#################################################")
    log.info("Number of Active BS: " + format(len(base_stations)))
    log.info("#################################################")

    # Instantiation of Gym environments for every BS
    log.debug("Initializing Gym Enviroment...")

    gym_envs = [gym.make('BS-v0', base_station=bs, monitoring_time_window=monitoring_time_window,
                         configfile=args.config, log_level=log_level) for bs in base_stations]

    agents = [Multi_Agent(drl_class, len(gym_envs[0].env.bs.Allocated_slices), env, config, config_name, log_level, env.bs.venodeb['slices'])
              for env in gym_envs]
    agents_data_size = compute_agent_data_size(agents, config.get('DRL', '_class_ML'))

    # clear or load pretrained models
    if clear_models:
        os.system('rm ../data/*')

    # Train the Agents for n_episodes
    for j in range(int(config.get('DRL', 'n_episodes'))):
        log.info("Federation episode = {} / {}".format(j, config.get('DRL', 'n_episodes')))

        json_payload = []
        for i in range(len(base_stations)):
            log.info("-------------------  BS {} -------------------".format(i + 1))
            agents[i].run()
            '''
            for a_ind, a in enumerate(agents[i].agents):
                filename_to_save = '../data/BS_{}_target_net_agent_{}.h5'.format(base_stations[i].ID, agents[i].slices_id[a_ind])
                if veNB_config[i]['federation'][a_ind]:
                    for tentative in range(2):
                        try:
                            a.train_network.save(filename_to_save)
                            break
                        except:
                            sleep(0.5) # wait while the file is being read by the other agents

                    filename_to_load = ['../data/BS_{}_target_net_agent_{}.h5'.format(vbs_id, agents[i].slices_id[a_ind]) for vbs_id in other_vbs]
                    for filename in filename_to_load:
                        if os.path.isfile(filename):
                            for tentative in range(2):
                                try:
                                    model_loaded = models.load_model(filename, compile=False)
                                    model_loaded.compile(loss=a._huber_loss, optimizer=Adam(learning_rate=(float(config.get('DRL', 'learning_rate')))))
                                    if np.mod((j+1), saving_period+5) == 0:  #todo: change1
                                        a.update_agent_network(inter_slice_federation(a.train_network, model_loaded))
                                    break
                                except:
                                    sleep(0.5) # wait while the file is being written by other agents
                a.update_agent_network(a.train_network.get_weights())         
                if np.mod((j+1), saving_period) == 0:
                    a.train_network.save(filename_to_save)
            '''
            data_size = agents_data_size[i]
            data = {
                "measurement": "model_size",
                "tags": {"ticker": base_stations[i].ID},
                "fields": {"size": data_size,
                           "fed_epoch": j}
            }
            json_payload.append(data)
        db_interactor.client.write_points(json_payload)

if __name__ == "__main__":
    main()


