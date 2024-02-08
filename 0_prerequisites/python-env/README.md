# Using the dedicated Python enviornment in terminal
To use the `Python` environment built for the course, you can simply activate.
First login to the `TALC` cluster using your terminal of choice:
```console
$ ssh your-UC-user-name@talc.ucalgary.ca
```
Use your UC password to login. Then, you need to load the list of HPC modules
first, and then proceed with activating the Python enviornment:
```console
$ module restore pbhm-mods
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/TALC/enci619.05_2024w/local/lib64/"
$ # activate the enviornment
$ source /work/TALC/enci619.05_2024w/local/python/pbhm-python-venv/bin/activate
```

# Using the dedicated Python environment in Jupyter Notebook as a Kernel
You need to first install the Python environment locally for your account with:
```console
$ module restore pbhm-mods
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/TALC/enci619.05_2024w/local/lib64/"
$ # activate the enviornment
$ source /work/TALC/enci619.05_2024w/local/python/pbhm-python-venv/bin/activate
$ python -m ipykernel install --user --name "pbhm-python-env"
```
After this one-time step, you can use this environment as a dedicated Kernel on your Jupyter Notebooks.

> [!NOTE]
> You only need to install the Python enviornment once to use in Jupyter Notebooks.

> [!CAUTION]
> You need to select the appropriate Python environment as a Kernel to run Notebooks for Assignment #3. After installing the Python environment as a Kernel, you can find the Kernel name in Jupyter Nobteooks under the `Kernel > Change kernel > pbhm-python-env` menu.
