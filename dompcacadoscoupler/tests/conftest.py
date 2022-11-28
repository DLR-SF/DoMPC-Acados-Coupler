import os

import pytest

# This is done so that you can set the _PYTEST_RAISE in your launch.json in VS
# Code. If you set the value of _PYTEST_RAISE to != 1, VS Code will also break
# on uncaught exceptions in your code using pytest. Otherwise, when setting the
# value to 1, the exceptions are caught by pytest and the debugger will not stop
# on an uncaught exceptions.
# From: https://stackoverflow.com/questions/62419998/how-can-i-get-pytest-to-not-catch-exceptions/62563106#62563106
# ! Deprecated.
# if os.getenv('_PYTEST_RAISE', '0') != '1':

#     @pytest.hookimpl(tryfirst=True)
#     def pytest_exception_interact(call):
#         raise call.excinfo.value

#     @pytest.hookimpl(tryfirst=True)
#     def pytest_internalerror(excinfo):
#         raise excinfo.value
