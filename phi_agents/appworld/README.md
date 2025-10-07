# AppWorld Interface

In order to keep `phi_agents/` dependencies separate from AppWorld (which relies on Pydantic v1), we've set up AppWorld to run as a separate server which runs in its own venv. By default, this venv is called `appworld-env`.

Notes

* Launching the server can be done as outlined in `server.py`.
    * Each server seems to read/write to the same `APPWORLD_ROOT`. Thus, it's recommended that unique episodes be created with unique experiment names, especially if they correspond to the same task ID.
* Interface for communicating with the AppWorld server is defined in `interface.py`.
* Example usage of AppWorld can be found in `tests/test_appworld_server.py`.


## Getting started

### Set up the appworld server

```bash
python -m virtualenv appworld-env
appworld-env/bin/pip install appworld
appworld-env/bin/appworld install
appworld-env/bin/appworld download data --root $APPWORLD_ROOT
```

Or using Conda

```bash
    conda create --name appworld python=3.11
    conda activate appworld   
    pip install appworld
    appworld install
    appworld download data --root $APPWORLD_ROOT
```

### Example scripts

* `scripts/run_appworld_inference.py`: Run inference (using a simple GPT-4o React agent) on AppWorld for a given dataset split.
* `scripts/visualize_appworld_queries.py`: Print the task queries for a given dataset split.
* `scripts/write_appworld_task_ids.py`: (Requires appworld installed) Save the task IDs for a given dataset split into a txt file. This is useful since we need access to dataset task IDs without importing appworld directly.
