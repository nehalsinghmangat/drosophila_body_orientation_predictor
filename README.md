# Installation
#### Backround: Data Source
The data in this paper is sourced from the experiments described and conducted in the paper [Plume-Tracking Behavior of Flying *Drosophila* Emerges from a Set of Distinct Sensory-Motor Reflexes](https://www.sciencedirect.com/science/article/pii/S0960982213015820) (F. van Breugel *et al.*, 2014). Three data sets are used, corresponding to measurements of flies flying in an artificial wind of 30 cm/s, 40 cm/s, and 60 cm/s. Each data set is itself composed of three subsets of data: flight trajectory data, body orientation data, and a key table to correlate the two for a given fly. All of this data is may be found at my [Dropbox](https://www.dropbox.com/scl/fo/igcpmck7s1tldkh9sumkm/h?rlkey=5n1rx7bk70gw8z2p3jcqr78l4&st=0kqw0da6&dl=0) with the following directory tree:

```
ExperimentalData
├── 30cms
│   ├── flight_trajectories_3d_HCS_odor_horizon_matched.h5
│   ├── body_orientations_HCS_odor_horizon_matched.h5
│   └── body_trajec_matches.h5
├── 40cms
│   ├── flight_trajectories_3d_HCS_odor_horizon_matched.h5
│   ├── body_orientations_HCS_odor_horizon_matched.h5
│   └── body_trajec_matches.h5
└── 60cms
    ├── flight_trajectories_3d_HCS_odor_horizon_matched.h5
    ├── body_orientations_HCS_odor_horizon_matched.h5
    └── body_trajec_matches.h5
``` 
### 1. Clone this repo:
``` bash
git clone https://github.com/nehalsinghmangat/drosophila_body_orientation_predictor.git
```
### 2. Download data:
Next, download `ExperimentalData` from my personal dropbox [here](https://www.dropbox.com/scl/fo/igcpmck7s1tldkh9sumkm/h?rlkey=5n1rx7bk70gw8z2p3jcqr78l4&st=4fqcasmi&dl=0) and clobber the existing `ExperimentalData` directory in the repo.

### 3. Build docker image:

If docker is not already installed on your system, please install  using the official [Docker documentation](https://docs.docker.com/engine/install/). Then navigate inside the `drosophila_body_orientation_predictor` repo and run the following command:
```bash
sudo docker build -t body_dros_venv .
```
This creates the docker image `body_dros_venv`, which you can view by running `sudo docker images ls`.
### 4. Run docker container
The docker container contains an instance of `jupyter-lab` whose kernel contains all dependencies required for the analysis. To create the container and run it, run the following command inside the repo:
```bash
sudo docker run -it -v .:/home --rm -p 8888:8888 body_dros_venv
```
This will start the notebook. The `-it` flag allows the user to interact with the container.  The `-v` flag mounts the `drosophila_body_orientation_predictor` host directory to the `/home` client directory; writing to one will write to the other, allowing the user to persist any data generated during the analysis. The `-p` flag maps the container port `8888` to the host port `8888` (this port must be available on your machine). To access the notebook, simply copy and paste the local URL generated in the container into your host's browser (the URL should begin with `http://127.0.0.1:8888`.) The `--rm` flag is discussed in step 6. 
### 5. Add MOSEK license
The `jovyan` directory is the default user directory for the `jupyter-lab` notebook. Because we use the MOSEK solver in `cvxpy`, the user must add a valid `mosek.lic` license to `jovyan/mosek`. If not, the user will be unable to correct the heading angles in the fly data.
### 6. Exiting safely
Because we ran the docker container with the `--rm` flag, upon exiting the container, the docker engine will automatically delete the container and its image. This will help keep the host machine's memory clean. If one wishes to persist the changes made in the docker container (for development purposes, say) then run step 4 without the `--rm` flag. This time, when you exit the container, it is stopped. To restart it and resume your session, run
```
sudo docker start --attach -i container_id
```
Note: you can run `sudo docker ps -a` to find the container id. 