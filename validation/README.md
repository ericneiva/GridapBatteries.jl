# Validation code (PyBaMM)

This directory contains the code to validate simple cases against PyBaMM.

## How to install?
These installation instructions assume you have Python installed (versions 3.9 to 3.11) and that you have also installed the `virtualenv` package which can be done by running
```bash
pip install virtualenv
```
Then you can install (assuming you run Linux or Mac) following these steps:
1. Create a virtual environment (this is strongly recommended to avoid clashes with the dependencies).
```bash
virtualenv env
```

2. Activate the virtual environment
```bash
source env/bin/activate
```
The virtual environment can later be deactivated (if needed) by running
```bash
deactivate
```

3. Install PyBaMM
```bash
pip install pybamm
```
Note that, currently, we need to install PyBaMM from a specific branch to get some necessary features not yet available in the develop branch, so we need to call
```bash
pip install git+https://github.com/pybamm-team/PyBaMM@issue-3115-shell-domains
```