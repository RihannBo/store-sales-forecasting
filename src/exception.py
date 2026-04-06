# Custom Exception Class
import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys): 
    
    _, _, exc_tb = error_detail.exc_info()  # returns tuple containing three values (type, value, traceback).
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    error_message = f"Error occured in python script name [{file_name}]\
                    line number [{exc_tb.tb_lineno}]\
                    error_message [{str(error)}]"
                    
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

# if __name__ == '__main__':
#     try:
#         logging.info("Logging has started")
#         a=1/0
#     except Exception as e:
#         custom_e = CustomException(e, sys)
#         logging.error(custom_e.error_message)
#         raise custom_e
        