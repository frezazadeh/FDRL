#!/bin/bash

echo "[1] Do you need help and you want to config the system?"
echo "[2] The system is config and you want generate tarffic, stop traffic generating or run the code?"
read -p "Please select the number (1, 2) :" USER_INPUT
if [[ $USER_INPUT == "1" ]]; then
     python3 amari_interactor/amari_client.py "help"
     read -p "Do you want to config the network? (y/n)" USER_INPUT2
     if [[ $USER_INPUT2 == "y" ]]; then
       read -p "How many users you want to activate? (1,2,3)" USER_INPUT3
       if [[ $USER_INPUT3 == "1" ]]; then
         python3 amari_interactor/amari_client.py "power_on $USER_INPUT3"
         echo "Your IP is ------------- 192.168.2.2 ----------------"
       elif [[ $USER_INPUT3 == "2" ]]; then
         python3 amari_interactor/amari_client.py "power_on 1"
         python3 amari_interactor/amari_client.py "power_on $USER_INPUT3"
         echo "Your IP is ------------- 192.168.2.2 & 192.168.2.6 ----------------"
       elif [[ $USER_INPUT3 == "3" ]]; then
         python3 amari_interactor/amari_client.py "power_on 1"
         python3 amari_interactor/amari_client.py "power_on 2"
         python3 amari_interactor/amari_client.py "power_on $USER_INPUT3"
         echo "Your IP is ------------- 192.168.2.2 & 192.168.2.6 & 192.168.2.10 ----------------"
       else
         echo "The system supports maximum 3 users"
       fi
       read -p "Do you config the compose file and ready to generate tarffic? (y/n)" USER_INPUT4
       if [[ $USER_INPUT4 == "y" ]]; then
         docker-compose -f docker-compose.yml build
         docker-compose up -d
       else
         echo "Tahnk you!"
       fi  
       read -p "Are you ready to config Cells/Slices? (y/n) -------- Please wait 30 seconds to sure for traffic generating" USER_INPUT5
       if [[ $USER_INPUT5 == "y" ]]; then
         python3 amari_interactor/amari_client.py "get_ue_ran"
         echo "--------------- Please check the ran_ue_id of users------------------"
         read -p "Are you ready for the next step? (y/n)" USER_INPUT6
         echo "-------------- You have $USER_INPUT3 active user(s) --------------"
         if [[ $USER_INPUT6 == "y" ]]; then
           for (( i=1 ; i<=$USER_INPUT3 ; i++ )); 
           do
             read -p "Enter ran_ue_id:" USER_INPUT7
             read -p "Select Cell/Slice ID (1, 2, 3):" USER_INPUT8
             if [[ $USER_INPUT8 == "1" ]]; then
               python3 amari_interactor/amari_client.py "handover $USER_INPUT7 1 524910"
             elif [[ $USER_INPUT8 == "2" ]]; then
               python3 amari_interactor/amari_client.py "handover $USER_INPUT7 2 528030"
             elif [[ $USER_INPUT8 == "3" ]]; then
               python3 amari_interactor/amari_client.py "handover $USER_INPUT7 3 530910"
             else
               echo "Tahnk you!"
             fi                
           done
           python3 amari_interactor/amari_client.py "get_ue_ran"
           read -p "The config is correct? (y/n)" USER_INPUT9
           if [[ $USER_INPUT9 == "y" ]]; then
             read -p "Do you want to run the code? (y/n)" USER_INPUT10
             if [[ $USER_INPUT4 == "y" ]]; then
               cd FederatedAgents/
               python3 main.py --config Config_BS/config.properties
             else
               echo "Tahnk you!"
             fi
           else
             echo "Please config Again!"
           fi  
         else
           echo "Tahnk you!"
         fi  
       else
         echo "Tahnk you!"
       fi       
     else
       echo "Thank You!"
     fi
elif [[ $USER_INPUT == "2" ]]; then
  echo "[1] The system is config and you want to generate tarffic?"
  echo "[2] Do you want to stop traffic generating??"
  echo "[3] The system is config and you want run the code?"
  read -p "Select a number (1, 2, 3)" USER_INPUT11
  if [[ $USER_INPUT11 == "1" ]]; then
    docker-compose -f docker-compose.yml build
    docker-compose up -d
    read -p "The system is config and you want run the code? (y/n)" USER_INPUT12
    if [[ $USER_INPUT11 == "1" ]]; then
      python3 FederatedAgents/main.py --config Config_BS/config.properties
    else
      echo "Thank you!"
    fi
  elif [[ $USER_INPUT11 == "2" ]]; then
    docker-compose -f docker-compose.yml down
  elif [[ $USER_INPUT11 == "3" ]]; then
     cd FederatedAgents/
     python3 main.py --config Config_BS/config.properties
  else
    echo "Thank you!"
  fi
else
  echo "Thank You!"
fi



