# Using the dedicated Python enviornment in terminal
To use the `Python` environment built for the course, you can simply activate it:
```console
$ source /work/TALC/enci619.05_2024w/local/python/pbhm-python-venv/bin/activate
```

# Using the dedicated Python environment in Jupyter Notebook as a Kernel
You need to first install the Python environment locally for your account with:
```console
$ source /work/TALC/enci619.05_2024w/local/python/pbhm-python-venv/bin/activate
$ python -m ipykernel install --user --name "pbhm-python-env"
```
After this one-time step, you can use this environment as a dedicated Kernel on your Jupyter Notebooks.

> [!NOTE]
> You only need to install the Python enviornment once to use in Jupyter Notebooks.

> [!CAUTION]
> You need to select the appropriate Python environment as a Kernel to run Notebooks for Assignment #3. After installing the Python environment as a Kernel, you can find the Kernel name in Jupyter Nobteooks under the `Kernel > Change kernel > pbhm-python-env` menu.
