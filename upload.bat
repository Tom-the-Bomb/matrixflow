@ECHO off

py -m build

py -m twine upload --config-file .pypirc --skip-existing dist/*