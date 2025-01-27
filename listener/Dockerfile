
FROM python:3.9

#Add the python file that we want to run in docker and define its location.
WORKDIR /
RUN apt-get update
RUN apt-get -y install iputils-ping
RUN apt-get -y install net-tools

COPY . /listener_docker/

WORKDIR /listener_docker

#instal the dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#Default port
#EXPOSE 8080

# Run the main
#CMD [ "python3" , "data_collector.py", "--config", "data_collector.config" ]
