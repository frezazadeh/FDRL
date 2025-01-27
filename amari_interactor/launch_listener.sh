#!/bin/bash

#./amari_api.js 10.1.14.249:9001
./amari_api.js localhost:9001
sleep 1
cd amari-exporter-linux-amd64
#./amarisoft-exporter-linux-amd64 -config config.yml
./amarisoft-exporter-linux-amd64 -config config_local.yml

