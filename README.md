## Getting Started
The **first time** building the dev env will take sometime since a lot of the libraries are large. After the first build, the dev env will
**spin up fast**.

### Dev Environment: Visual Studio code (vscode) & Docker (Recommended) **CUDA** 12.6

Install `docker` and `docker-compose` and have the **docker application running** before attempting to run commands with `docker`, `docker-compose`, or spinning up a development environment

Make sure to clone this repo an `cd` into the **root directory** of this project before running any commands

This method requires vscode to be installed as this method will attach to running local development containers using vscode. Furthermore, in vscode, two extension are **needed** `Dev Containers` and the `Docker` extension. These can be found and installed in vscode under the *Extensions* tab. Specificity, Dev Containers will allow us to to attach a vscode instance to a running container. Dev Containers can also spin up the containers too if they are not running already when attaching. 

First, open vscode in the root directory of this project. This is the directory where `docker-compose.dev.yml` is in.

Then open the Command Palette under View

the shortcut is as follows

Windows: `f1`

Mac: `Shift` + `Command` + `P`


1. Run Dev Containers: Open Folder in Container... .devcontainer

2. VS Code will then start up the container and connect this window to it

### Dev Environment: Docker
Manually build and attach with vscode

1. (in project root dir) `docker compose up --detach`
2. In vscode in Containers extension tab attach vscode code to project container

### Dev Environment: venv
One can use venv or other methods too. requirements.txt can be used with torch==2.3.0 torchvision==0.18.0, to align with the
docker dev env if needed.