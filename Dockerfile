# jupyter in container
FROM  jupyter/base-notebook:python-3.10.10

# non-interactive mode for automated installations
ARG DEBIAN_FRONTEND=noninteractive

# working directory in container
WORKDIR /home

# copy relevant folders and files
# COPY ExperimentalData /home/jovyan/work/ExperimentalData
# COPY Notebooks /home/jovyan/work/Notebooks
COPY requirements.txt /home/

USER root

RUN apt update && \
    apt upgrade -y && \
    apt install git -y && \
    pip install -r /home/requirements.txt && \
    pip install pychebfun
