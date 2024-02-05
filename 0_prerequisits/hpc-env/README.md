# Saving list of modules for `TALC`

To save the list of modules, copy the [saved list of modules](./pbhm-mods) to the `$HOME/.module` directory.
```console
$ mkdir $HOME/.module
$ cp pbhm-mods $HOME/.module
```

Whenever you need the modules in the terminal environment, you can restore them using:
```console
$ module restore pbhm-mods
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/TALC/enci619.05_2024w/local/lib64"
```

> [!CAUTION]
> Please rememebr that you need to reload the list of modules whenever you want to use them in the terminal environment.

> [!NOTE]
> The loading of `HPC` modules are already taken care of in Assignment #3.
