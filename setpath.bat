# Set the directory where Falcor binaries and Python module are located
$FALCOR_DIR = "D:\NeuralFalcor\neuralrender2\build\windows-vs2022\bin\Release"

# Update the PATH environment variable
$env:PATH = "$FALCOR_DIR;$env:PATH"

# Update the PYTHONPATH environment variable to include the Falcor Python directory
$env:PYTHONPATH = "$FALCOR_DIR\python;$env:PYTHONPATH"

# Verify the environment variables (Optional step for troubleshooting)
echo "FALCOR_DIR: $FALCOR_DIR"
echo "PATH: $env:PATH"
echo "PYTHONPATH: $env:PYTHONPATH"

# Start Python and import Falcor
python
>>> import falcor
