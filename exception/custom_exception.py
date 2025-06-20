class ErrorMessageException(Exception):
    def __init__(self, code, params=None):
        self.code = code
        self.params = params