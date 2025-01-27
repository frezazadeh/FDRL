# -----------------------------------------------------------
# Amarisoft client for the Decision Engine
#
# email: sbarrachina@cttc.es
# -----------------------------------------------------------

import asyncio
import sys
import time
import websockets   # Install locally through command pip install "websockets==8.1"
import json
import pprint
import os
from datetime import datetime

TARGET_ENB = "10.1.14.249:9001"  # Socket for of Amarisoft gNB (or eNB)
TARGET_MME = "10.1.14.249:9000"  # Socket for of Amarisoft AMF (or MME)
TARGET_UE = "10.1.14.250:9002"  # Socket for of Amarisoft Simbox

pp = pprint.PrettyPrinter(indent=4)

async def amarisoft_api_request(target, msg):
    """
    amarisoft_api_request performs a call to an Amarisoft API through WebSocket
    
    :param target: target (endpoint) of the API
    :param msg: message to be sent to the API
    :return json: API response in JSON format
    """ 

    uri = "ws://" + target

    async with websockets.connect(uri, origin="Test") as websocket:

        ready = await websocket.recv()
        await websocket.send(msg)   
        rsp = await websocket.recv()
        # pp.pprint(json.loads(rsp))

        return json.loads(rsp)

def perform_api_call(target, msg):
    """
    perform_api_call performs a call to an Amarisoft API (Callbox or Simbox)

    :param msg: message to be sent to the API
    :return json_response: API response in JSON format
    """ 
    json_response = None
    while json_response is None:
        try:
            # connect         
            json_response = asyncio.run(amarisoft_api_request(target, msg))                            
            return json_response
        except:
            print("EXCEPTION: something went wrong when connecting to Amarisoft API. Retrying...")
            time.sleep(2)

def parse_command_to_msg(command):
    """
    parse_command_to_msg parses command entered by console to Amarisoft API message

    :param command: command entered by console
    :return target: API endpoint
    :return msg: API message corresponding to the entered command
    """ 
    
    split_command = command.split()
    action = split_command[0]

    if action == "get_ue_sim":
        msg = '{"message":"ue_get", "stats":true,}'
        target = TARGET_UE
        return target, msg

    if action == "get_ue_ran":
        msg = '{"message":"ue_get", "stats":true,}'
        target = TARGET_ENB
        return target, msg

    if action == "get_ue_core":
        msg = '{"message":"ue_get", "stats":true,}'
        target = TARGET_MME
        return target, msg

    if action == "handover":
        ran_ue_id = split_command[1]
        cell_pci = split_command[2]
        ssb_nr_arfcn = split_command[3]
        msg = '{"message":"handover", "ran_ue_id": ' + ran_ue_id + ', "pci": ' + cell_pci + ', "ssb_nr_arfcn": ' + ssb_nr_arfcn + '}'
        target = TARGET_ENB
        return target, msg
    
    if action == "power_on" or action == "power_off":
        ue_id = split_command[1]
        msg = '{"message": "' + action + '", "ue_id": ' + ue_id + '}'
        target = TARGET_UE
        return target, msg

    

def main():

    # print("---------------------") 
    # print("Amarisoft API client")
    # print("---------------------") 

    amari_command = ""
    #print("Argument List:", str(sys.argv))
    if (sys.argv[1] == "help" or sys.argv[1]=="h"):
        print(("Use 'python3 amari_client.py \"<AMARISOFT_COMMAND>\"', where <AMARISOFT_COMMAND> can be:"))
        print(" get_ue_sim\t gets list of UEs with details from Simbox side on their ue_id, dl_bitrate, imsi, etc.")
        print(" get_ue_ran\t gets list of UEs with details from RAN side on their cell, MCS, etc.")
        print(" get_ue_core\t gets list of UEs with details from CORE side on their ran_ue_id, imsi, etc.")
        print(" power_on UE_ID\t powers ON the UE with Simbox's UE_ID.")
        print(" power_off UE\t powers OFF th UE with Simbox's UE_ID.")
        print(" handover RAN_UE_ID CELL_CPI SSB_NR_ARFCN\t performs handover of UE with Callbox's RAN_UE_ID to cell with CELL_CPI and SSB NR ARFCN SSB_NR_ARFCN.")
        
        sys.exit()

    # Examples for handover calls (assuming RAN_UE_ID is 10):
    # To cell 1: python amari_client.py "handover 10 1 524910"
    # To cell 2: python amari_client.py "handover 10 2 528030"
    # To cell 3: python amari_client.py "handover 10 3 530910"

    if len(sys.argv) == 2:
        amari_command = sys.argv[1]
        #print("Entered Amarisoft command: " + amari_command)
    else:
        sys.exit("Missing Amarisoft command! ---> Use 'python3 amari_client.py \"<AMARISOFT_COMMAND>\"'")
    print("----------------------------------------------------") 

    # Parse command to API message
    [target,msg]=parse_command_to_msg(amari_command)
    print("Call:")
    print(msg)

    # Call API
    json_response= perform_api_call(target,msg)
    print("Response:")
    pp.pprint(json_response)

 
if __name__ == "__main__":
    main()
