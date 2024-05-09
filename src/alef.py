from sys import stderr
import argparse
import logging

ELEMENT_NAME = "macrophage"

# Create the logger
handler = logging.StreamHandler(stream=stderr)
logger = logging.getLogger()

# Set the formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
handler.setFormatter(formatter)

# Set the handler
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Parse the required command-line arguments
parser = argparse.ArgumentParser(description='A simple program to demonstrate argparse usage.')
parser.add_argument('-I', type=str, help='Input image absolute or relative path', required=True)
parser.add_argument('-O', type=str, help='(Unused) this model does not accept output', required=False)

# Parse the arguments
args = parser.parse_args()
filepath = args.I

# Import the analysis model 'analyze' function
from model import analyze

# Call the model
try:
    # Get the results
    results = analyze(filepath=filepath)
    logger.info(f"done, detected a total of ({len(results)}) macrophages")
    
    # Print the results in ALEF format
    print(f'{ELEMENT_NAME}=', end='')
    
    for element in results:
        print(element["x"], end=':')
        print(element["y"], end=',')
    
    print(f';', end='')
    print()
    
# Get the failed execution
except Exception as e:
    logger.fatal(e)