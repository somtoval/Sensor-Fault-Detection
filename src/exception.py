import sys

# This function takes in the exception error that occured and the sys object that was taken at the point of the exception
def error_message_detail(error, error_detail: sys):
    # This function from the error_detail module is used to retrieve information about the current exception that is being handled. It returns a tuple containing three elements: the exception type, the exception object, and the traceback.
    # Here the excetption traceback is stored in a variable exc_tb
    _, _, exc_tb = error_detail.exc_info()

    # The code you provided is written in Python and is used to extract the filename associated with the code where the exception occurred. It's a part of error handling or exception handling mechanism. Here's what the code does: exc_tb: This variable holds the traceback object, which was likely obtained using the exc_info() function from the traceback module. The traceback object contains information about the call stack at the point where the exception occurred. tb_frame: This is an attribute of the traceback object. It represents the frame (execution context) where the exception occurred. f_code: This is an attribute of the frame object. It represents the code object associated with the frame, which contains information about the code being executed. co_filename: This is an attribute of the code object. It represents the filename of the source code associated with the code object, i.e., the file where the code is defined.
    # By executing this code, file_name will store the filename of the source code where the exception was raised.
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Asigning a comprehensive error message consisting of line number, file name and The variable error is assumed to contain the exception object, and str(error) converts it into a string representation.
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    # Returning the error_message variable containing a comprehensive message
    return error_message

# Creating a custom exception that inherits from python built-in exception class
class CustomException(Exception):
    # Accepting exception message and sys object of the exception as input arguments using init construction
    def __init__(self, error_message, error_detail: sys):
        """
        :param error_message: error message in string format
        """
        # Passing the error message to the inherited exception class
        super().__init__(error_message)

        # We then pass our exception message and exception sys object as the value of error message
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    # Returning the string format of error_message as it is a variable function(just like object of a class)
    def __str__(self):
        return self.error_message
