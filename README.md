# Table of Contents

1. [Introduction](#introduction)
2. [About the Project](#about-the-project)
3. [Dataset Information](#dataset-information)
4. [Proof of Cloud Deploy](#proof-of-cloud-deploy)
5. [Files & Folders](#files--folders)
6. [Setup Prerequisites](#setup-prerequisites)
7. [Setting up Local Environment](#setting-up-local-environment)
8. [SaturnCloud Setup](#saturncloud-setup)
9. [Downloading Data](#downloading-data)
10. [Model Training & Deploy](#model-training--deploying)
11. [Testing Model](#testing-model)
12. [Cloud Deployment](#cloud-deploying)

# Introduction

This is the second capstone project I built for the Machine Learning Zoomcamp by Alexey Grigorev. Learn more about the free bootcamp by following any of the links below. 

- https://mlzoomcamp.com
- https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp

For this project, we were required to pick out a dataset of our choosing, and then make and train a ML model using it. We were also required to deploy it either to a cloud service or to our local systems. 

I additonally built a script that uses command line utility to easily query either deploy of the model. 

# About the Project

I wanted to do something a bit more visually appealing to work with than... *kitchenware* after the last project. A bit of browsing on Kaggle led me to a dataset filled with stunning landscapes, broken down into five categories. 

Is it practical? Nope, not even close. Was it fun to see pretty landscapes as I tore my hair out debugging instead of *forks and spoons*? Absolutely. 10/10 would recommend landscapes over forks and spoons. 

I built a model that classifies landscape images into five different categories: Coast, Desert, Forest, Glacier, and Mountain. All it needs is the URL to a publicly hosted .JPG image, and it will classify it into one of these categories and output its prediction with ~96% accuracy. 

# Dataset Information

The data was obtained from this publicly available [Kaggle dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images).

From source, the data is split into three folders for Training, Validation, and Testing. Each of these is further split into sub-folders for each category, which contain the images. 

By source the data was also in several nested folders. I reorganise these for ease of use when training the model, and all the steps I took are detailed in the appropriate section below. 

# Proof of Cloud Deploy

Just in case the cloud deploy is not live for any reason, the proof folder contains screenshots of the testing script querying the cloud-hosted model with different images and the model returning predictions.

# Files & Folders

There are multiple files and folders provided in the main repository. Below, I organise them loosely, but each of the files and their purpose has been explained extensively in the appropriate sections. 

1. Setup
    - `requirements.txt` - for setting up the virtual environment. **NOTE: The setup of this virtual environment could cause conflicts on non-Windows setups. See ["Saturn Cloud Setup"](#saturn-cloud-setup) below for a quick guide on running the repository on Saturn Cloud instead.**

2. EDA & Model Experimentation
    - `01_eda.ipynb` - notebook containing data analysis. 

    - `02_model1_train.ipynb` - notebook containing the training and tuning of the first model. 

    - `03_model2_train.ipynb` - notebook containing the training and tuning of the second model. 

3. Picking & Saving Best Model
    - `04_model_test.ipynb` - notebook in which we test both models against testing data and pick the best one to deploy. 

    - `train.py` - isolated Python script to train the best model and save it as a ONNX model.

4. Serving the Model
    - `landscape_model_resnet50.onnx` - ONNX model file to deploy without the need of running the training script. 

    - `predict.py` - helper functions to process the image file, obtain the predictions, and output the final result. 

    - `service.py` - Flask program to serve the model.

    - `Dockerfile` - Dockerfile to build and package the Flask service and the helper functions.

    - `docker_req.txt` - a requirements file containing just the dependecies needed within the Docker container. Excludes libraries used in training, saving, and querying the model. Used only in Docker container.

5. Getting Results from Model
    - `query.py` - isolated Python script that can make requests to the model through the command line and return the output.

# Setup Prerequisites 

All of the setup instructions below assume you have a working Python install ready. It also assumes you have Docker installed.

If you do not have any of these, please follow the respective guides to install and set them up before following any steps below.

- [Python Installation](https://docs.python.org/3/using/index.html)
- [Docker Installation](https://docs.docker.com/engine/install/)
- [Setting up Docker and Windows Linux Subsystem - For Windows Users](https://andrewlock.net/installing-docker-desktop-for-windows/)

# Setting up Local Environment

__Any code instructions listed throughout this guide are for a Windows system. Where possible, links to tutorials detailing the same or similar steps in other OS have been provided.__

To run this repository on your local system, you will need to setup a Python virtual environment. I use the built-in Python `venv` for this. [Python documentation](https://docs.python.org/3/library/venv.html) has an extensive guide on creating these environments. 

1. Create a new folder and navigate to it. 

2. Create a virtual environment. 
    ```
    python -m venv /path/to/venv/directory
    ```

3. Activate the new virtual environment.
    
    ```
    .\Scripts\Activate.ps1
    ```
    Make sure you are in the virtual environment directory before running this command.

4. Clone the project repository.

    ```
    git clone https://github.com/AaishaMuhammad/mlzoomcamp_capstone2
    ```

After ensuring you are navigated to the directory containing all the files,

7. Install all required packages.
    
    ```
    pip install -r requirements.txt
    ```

<br>

# SaturnCloud Setup

Due to the fact that training and tuning the models needs GPU augmentation for better speed, I would recommend using a SaturnCloud setup if you want to re-run the training scripts. SaturnCloud gives 30 free hours of GPU usage a month. If you do use SaturnCloud, here a quick setup guide. 

1. Create SaturnCloud Resource

    I use the PyTorch template for ease. Make sure whatever resource you pick has GPU. 

2. Configure the Resource

    After the resource has been created, go to 'Edit' and change the following options:

    - **Disk Space**: The resource needs more than the default disk space. Set it to 100GiB which is within the free plan.
    - **Environemnet -> Pip**: Scroll down to the environment section and click on the tab that says 'Pip'. You will see a field that says something similar to this: `torch dask-pytorch-ddp`. Add `kaggle seaborn` to the end of this list. We will need this to use the Kaggle API to download the data, and Seaborn is used in the EDA notebook.

3. Install Other Dependencies 

    Depending on which resource template you used, you may have to aditionally run: 

    ```
    pip install fastai matplotlib
    ```

    However some resources already have these preinstalled. 

# Downloading Data

0. Set up Kaggle Api

    If you don't already have it configured, you will need to set up the Kaggle API. Follow the instructions [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) to set it up on a local machine. 

    For SaturnCloud, you need to copy the entire contents of `kaggle.json` as a Secret with the name 'kaggle' to your account first. Then navigate to your new resource and go to the 'Secrets' section in the menu. 

    Add the 'kaggle' secret as a 'Secret File' with the following path: 

    ```
    /home/jovyan/.kaggle/kaggle.json
    ```
    
    This will now authenticate the Kaggle API. 

**Make sure the Git repo is cloned before following this step to ensure the folder structure is in place.**

1. Create Data Folder

    Navigate to the `mlzoomcamp_capstone2` project folder. Create a new folder 'data' and navigate into it. 

    Run the following command: 

    ```
    kaggle datasets download -d utkarshsaxenadn/landscape-recognition-image-dataset-12k-images
    ```

2. Unzip Files
    
    The Kaggle API downloads the data as a zipped folder. Extract all the files and save them into the 'data' folder.

3. Move Files Into `data` Folder

    Navigate into the unzipped folder. Copy or move the three folders `Training Data`, `Validation Data`, and `Testing Data` into the main `data` folder. 

    The final folder tree should look as such: 

    --- data

        --- Testing Data

        --- Training Data

        --- Validation Data

    This data structure has to be the same in order for the Python scripts and notebooks to run correctly. Any extra unzipped folders or files do not matter and can be ignored as long as these three folders have the same structure. 

# Model Training & Deploying

Make sure you have followed all the previous steps before attempting these. You will need to be in the activated virtual environment and have all dependencies installed for this part to work.

1. Run the training script.

    ```
    python train.py
    ```

    *This step is **optional**. A model file has been provided in the repository and will be used instead if the training script is not run.*

2. Build the Docker container.

    ```
    docker build -t landscape_recognition . 
    ```

This part may take some time to run. After it has finished,

3. Run the docker container.

    ```
    docker run -it --rm -p 8080:8080 landscape_recognition 
    ```

Congratulations! The model is now up and running on your local system.

# Testing Model

The model can be tested using the provided script directly through the command line. 

1. Customise the query.

    Open the `query.py` file in your code editor. At the top there are two predefined URL variables. The one for local deploy will be commented out - uncomment it, and comment the definition below that tests the cloud deploy. 

2. Run the query

    ```
    python query.py https://www.shutterstock.com/image-photo/glaciers-icebergs-antarctica-very-south-260nw-1935877102.jpg
    ```

    Or, replace the url in the above command with any .JPG image of a landscape of your choice. 

<br>

__NOTE: Make sure there is nothing ahead of the '.jpg' part of the URL. URL parameters may cause errors. No formats other than .jpg work with the model at present.__

And that is it! The output will be printed onto the terminal. 

<br>

# Cloud Deploying

In a real-world production scenario, we will have to deploy the model to some cloud service to allow it to be queried from external services. Right now, we can only query it from another app running either on our local system or at least on the same network. Once it is deployed to the cloud, external apps can also access it - like my query script app works on my cloud deploy. 

This guide will take you over the steps required to deploy to Google Cloud Run. Before following these steps, make sure you have an active Google Cloud account. You will need to have activated and verified Billing Options as well. 

This will also require an active installation of Gcloud CLI. A detailed guide on installing and running this can be found on [Google Documentation](https://cloud.google.com/sdk/docs/install). 

__This model may be heavy on resources - running it on Google Cloud Run may incur minor costs.__

0. Prerequisites

    Before attempting any of the steps below, make sure you have finished all the previous steps and have a ready Docker image. If your container is currently running on the local system, you may deactivate it - it is not needed for this section. 

    To check if your Docker image is ready, run:

    ```
    docker images
    ```

    You should see a docker image similar to `landscape_recognition:'unique-tag'` in the list. 


1. Create a project in Google Cloud.

    Start by navigating to the [Google Cloud console](https://console.cloud.google.com/projectselector2/home/dashboard) and creating a new project. Make sure billing options are activated for this project. 


2. Enable Artifact Registry.

    We need the artifact registry to get our Docker container over to Google Cloud Run. Enable the API by following [this link](https://console.cloud.google.com/apis/enableflow?apiid=artifactregistry.googleapis.com) and completing the steps. 

_The user may require special permissions to run Docker. If at any point you run into permission issues, try again from an administrative user._


3. Create artifact repository.

    Navigate to the Artifact Registry and open the Repositories page. 

    Create a new repository and name it `'landscape-recognition'`. Make sure you have the format configured to `'Docker'` and choose a region. I used `'us-central1'` for my deploy. 


4. Set up Docker authentication.

    *If you have pushed Docker images to Google Cloud before, this should be already configured and you may skip this step.*

    You need to configure Docker to use Google Cloud to authenticate requests when pushing or pulling Docker images. Run the following command in your terminal to update the Docker configuration:

    ```
    gcloud auth configure-docker us-central1-docker.pkg.dev
    ```

    If you used another region and not 'us-central1', change that part of the command to match your region.


5. Tag Docker image.

    Before we can push the Docker image, we need to tag it so we can push it to a specific location:

    ```
    docker tag landscape_recognition:'your_unique_tag'  us-central1-docker.pkg.dev/'your_project_id'/landscape-recognition/landscape_recognition:latest
    ```
    
    Use the above command after changing `'your_unique_tag'` to the tag shown when you ran `'docker images'`. Also change `'your_project_id'` to whatever ID you specified when creating your Google Cloud project. 

    If you used another region and not 'us-central1', change that part of the command to match your region. 


6. Push the Docker image.

    Now we can push our Docker image onto the Artifact Registry. Run:

    ```
    docker push us-central1-docker.pkg.dev/'your_project_id'/landscape-recognition/landscape_recognition:latest
    ```

    Making sure to replace `'your_project_id'` with your project ID, and also the region if 'us-central1' was not used. 

_If you would like a more in-depth reading about the above, or need to find help for a different OS, consult [these docs](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images)._

<br>

Now that we have pushed the Docker image onto Artifact Registry, it is available for us to use in a Cloud Run project. 

7. Create Google Cloud Run service.

    Navigate to the Google Cloud Run [dashboard](https://console.cloud.google.com/run). Click on __Create Service__. 

    On the left side, select the option to 'Deploy one revision from an existing container image'. Navigate to and select the Docker image we just pushed.


8. Specify details.
    
    Fill in the rest of the form with required detials, such as project name and region. Use the same region and pick a convenient name like 'landscape-recognition' as you cannot change this part. 

    Set CPU allocation and autoscaling as needed. Specify disk space to 4GiB.


9. Set authentication.
    
    Make sure you pick the authentication option, __Allow unauthenticated invocations__. This will allow us to query it with the `query.py` script.


10. Create service.

    Once you've set up everything, click Create and wait for the deployment to finish. This can take a moment to process everything. 


11. Configure `query.py`.

    After the service is deployed, you will see a displayed URL in the header. Copy this URL. 
    
    Navigate to `query.py`. Remove the URL in the cloud deploy variable and paste your deploy's link in place. Save the file and run,

    ```
    python query.py `image-url-here`
    ```

    The script will now query your cloud deploy of the model. 


12. Delete Docker image from repository.

    After the Google Cloud Run service is deployed, we can safely delete the image from the Artifact Registry.

    Navigate back to the Registry and from there to Repositories. Select the 'landscape-recognition' repository and click on 'Delete'. Make sure you confirm the deletion and wait for it to finish deleting. 
