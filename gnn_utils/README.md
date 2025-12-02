## 1. Dataset Generation

File: generate_graph_dataset.ipynb

This notebook creates the graph-based dataset used for GraphGPS.

## 2. Hyperparameter Search (GraphGPS)
File: gnn_hyperparam_search.py

This script performs a Ray Tune hyperparameter search and logs all runs to MLflow.

Before running: Start Ray + MLflow servers:
# Start Ray cluster
ray start --head --dashboard-host 127.0.0.1

# Start MLflow tracking server
mlflow server --host 127.0.0.1 --port 8080

Run the hyperparameter search: python gnn_hyperparam_search.py

View MLflow experiment dashboard Open: http://127.0.0.1:8080

## 3. Train the Final GraphGPS Model
File: gnn_recommender.py

Use this script with the best hyperparameters to train the final model, based on the best Ray Tune run.

Only MLflow must be running for tracking: mlflow server --host 127.0.0.1 --port 8080

Then train the model: python gnn_recommender.py

The first build of the development environment may take some time due to library size. After that, the environment starts quickly.

### Recommended: VS Code Dev Containers (Docker, CUDA 12.6)

1. Install **Docker** and **docker-compose**.
2. Ensure Docker Desktop is running.
3. Install the following VS Code extensions:
   - **Dev Containers**
   - **Docker**
4. Open VS Code in the project root (where `docker-compose.dev.yml` is located).
5. Open Command Palette:
   - Windows: `F1`
   - Mac: `Shift + Cmd + P`
6. Select: **Dev Containers: Open Folder in Container…**
7. Choose the `.devcontainer` folder.

VS Code will then build and attach to the development container.

Then attach to the container via the Docker tab in VS Code.

### venv Alternative

A Python virtual environment can also be used.  
Recommended versions to match the container:
