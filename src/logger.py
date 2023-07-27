# import logging
# import os
# from datetime import datetime

# # datetime.now() is a function that returns the current date and time as a datetime object, .strftime('%m_%d_%Y_%H_%M_%S'): is a method that is used to format the datetime object as a string according to the provided format, e.g: 07_24_2023_15_30_45 (assuming the current date is July 24, 2023)
# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# # Line of code joins the current working directory, "logs" and the LOG_FILE as a directory
# logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# # Here the logs_path directory is created using os.makedirs() and using exist_ok=True means that if the directory already exists, the function will not raise an error. If the directory already exists, the function will silently continue without attempting to recreate the directory.
# os.makedirs(logs_path, exist_ok=True)

# # We created the logfile file path by joining the logs_path and LOG_FILE so that our logs directory will be like this "logs\07_06_2023_16_18_03.log\07_06_2023_16_18_03.log"
# LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# # Creating an instance of the python logger class and configuring it
# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO,
# )

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
