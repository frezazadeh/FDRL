# Dockerfile for a test machine with traffic utils
# **Build command**: $ docker build --tag test_traffic:latest .

## Use baseimage-docker's init system.
FROM phusion/baseimage:master
CMD ["/sbin/my_init"]

## Install common utils and libs
RUN install_clean cmake build-essential libpcap0.8 libpcap0.8-dev iperf3 iperf git unzip iputils-* net-tools iproute2 tcpdump stress traceroute

## Clone and build MGEN (traffic generator util)
WORKDIR /opt
RUN git clone https://github.com/USNavalResearchLaboratory/mgen.git
WORKDIR /opt/mgen
RUN git clone https://github.com/USNavalResearchLaboratory/protolib.git
WORKDIR /opt/mgen/makefiles
RUN make -f Makefile.linux


## Install python and dependencies
RUN apt-get update
RUN apt-get install -y build-essential python3.8 python3-pip nano
RUN python3 -m pip install --upgrade pip
RUN pip3 install pandas influxdb psutil configparser datetime

COPY ./traffic_monitoring.py /opt/mgen/makefiles
COPY ./traffic_generator.config /opt/mgen/makefiles


# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
