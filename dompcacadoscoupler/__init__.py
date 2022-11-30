# Read https://timothybramlett.com/How_to_create_a_Python_Package_with___init__py.html how to add imports.
# By adding imports in this file you can more easily import the classes and functions from this package into other Python modules.
# Example:
# from .your_module import your_class
from sys import platform
from warnings import warn

import colorama
from termcolor import colored

colorama.init()
if platform == "linux" or platform == "linux2":
    pass
elif platform == "darwin":
    # OS X
    warn(colored(f'Acados may not work on an OS X platform.', 'yellow'),
         stacklevel=2)
elif platform == "win32":
    # Windows
    warn(colored(f'Acados may not work on a Windows platform.', 'yellow'),
         stacklevel=2)
