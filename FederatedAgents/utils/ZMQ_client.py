import time
import zmq
import msgpack
import numpy as np
import json

SERVER = 0
CLIENT = 1

if SERVER == 1:
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    # socket.setsockopt( zmq.RCVTIMEO, 2000 ) # milliseconds
    socket.bind("tcp://*:5555")

    print("Listening")
    while True:
        try:
            message = socket.recv_string(encoding='utf-8', flags=zmq.NOBLOCK)  # socket.recv()
            print("MESSAGE:" + message)
            socket.send("REPLY")

        except zmq.Again as e:
            time.sleep(1)
            a = 2  # do nothing
            # print "No message received yet"

        except Exception as err:
            a = 2  # do nothing
            # print(Exception)
            # print("Error" + format(err))
            pass

        #  Send reply back to client
        # socket.send(b"REPLY")

if CLIENT == 1:
    #
    #     prbs = np.arange(5,100,5)
    #     context = zmq.Context()
    #     socket = context.socket(zmq.DEALER)
    #     socket.connect("tcp://*:6666")
    #     print("Connecting to srsLTE scheduler...")
    #
    #     for i in range(len(prbs)):
    #         socket.send(format(prbs[i]).encode('utf-8'),flags=zmq.NOBLOCK)
    #         print("Sending ..." + format(prbs[i]))
    #         time.sleep(3)
    #     print("Done")

    context = zmq.Context(1)
    # print("Connecting to hello world server")
    socket = context.socket(zmq.REQ) #PUSH
    # socket.bind("tcp://*:5556")  # Connect to external port of docker

    #socket.connect("tcp://localhost:5556") # Connect to external port of container
    socket.connect("tcp://10.10.244.69:5000")  # Connect to external port of container

    #  Do 10 requests, waiting each time for a response
    prb_list = [1, 10, 20, 30, 40, 50]
    for request in prb_list:

        # message = {"slice1": request, "slice2": request}
        # myjson = json.dumps(message)

        message2 = format(request) + ',' + format(request) # downlink, uplink

        print("Sending request " + format(message2))
        socket.send_string(message2, zmq.NOBLOCK)
        time.sleep(5)

        # Get the reply.
        message = socket.recv()
        print("Received reply %s [ %s ]" % (request, message))
        time.sleep(1)

