import json
from model import analyze, ELEMENT_NAME
from sys import stderr
import argparse
import logging


def main():
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
    parser = argparse.ArgumentParser(
        description='Leishmaniasis macrophages identification analysis model')

    parser.add_argument(
        '--alef-in',
        '-i',
        dest="input",
        type=str,
        help='Input image absolute or relative path',
        required=True,
    )

    parser.add_argument(
        '--alef-out',
        dest="output",
        type=str,
        help='(Unused) this model does not accept output',
        required=False,
    )

    # Parse the arguments
    args = parser.parse_args()
    filepath = args.input

    # Import the analysis model 'analyze' function

    # Call the model
    try:
        # Get the results
        results: dict = analyze(filepath=filepath)
        logger.info(
            f"done, detected a total of ({len(results[ELEMENT_NAME])}) macrophages")

        # JSON output
        print(json.dumps(results))

    # Get the failed execution
    except Exception as e:
        logger.fatal(e)


if __name__ == '__main__':
    main()
