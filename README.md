# DoMpc Acados Coupler

## Compatibility between DoMPC and Acados

Currently not supported functionalities are:

MPC
- The rterm is not fully supported
- Only bound constraints can be considered
- Robust MPC

MHE
- Not implemented yet

Currently not tested functionalities are:
- Visualization via the Graphics module

## Installation

For using this package you need to have Acados and the Acados Python package installed.
For installing these see the [Acados documenation](https://docs.acados.org/).

By installing the project as a package you can import the modules within the package in other Python modules by using:

> `import dompcacadoscoupler`

To install the project just type
> `pip install .`

in your command line tool. The `.` stands for the current directory. Thus, you must be in the project root directory.

## Examples

Example for how to use the package can be found in the `tests` folder.

## Developer Instructions

Install in "develop" or "editable" mode (change directory to the root folder with the setup.py in it and execute via the anaconda prompt for example):

> `pip install -e . -r requirements_dev.txt`

It puts a link (actually *.pth files) into the python installation to your code, so that your package is installed, but any changes will immediately take effect.
Furthermore, it installs some useful libraries for developing, which are defined in the `requirements_dev.txt` file.

In order to test the package execute

> `pytest --pyargs dompcacadoscoupler`

## Rationale Project Structure

- The `tests` folder needs a `__init__.py` file, otherwise VS Code can not execute the tests.
