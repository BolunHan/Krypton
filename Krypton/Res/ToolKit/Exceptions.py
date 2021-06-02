__all__ = ['Exceptions']


class CodedExceptions(object):
    def __init__(self):

        self.exceptions = {
            8001: self.Deprecation,
            8002: self.IterationEnd,
            8003: self.Timeout,
            8004: self.InvalidKey,
            8005: self.InvalidType,
            8006: self.InvalidValue,
            8007: self.LoginFailure,
            8008: self.NextTry,
            8009: self.MaxRetry,
            8010: self.CacheError,
        }

    def __getitem__(self, items):
        return self.exceptions.__getitem__(items)

    class Deprecation(Exception):
        def __init__(self, error_message: str, error_code: int = 8001, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class IterationEnd(StopIteration):
        def __init__(self, error_message: str, error_code: int = 8002, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class Timeout(TimeoutError):
        def __init__(self, error_message: str, error_code: int = 8003, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class InvalidKey(KeyError):
        def __init__(self, error_message: str, error_code: int = 8004, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class InvalidType(TypeError):
        def __init__(self, error_message: str, error_code: int = 8005, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class InvalidValue(ValueError):
        def __init__(self, error_message: str, error_code: int = 8006, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class LoginFailure(ConnectionError):
        def __init__(self, error_message: str, error_code: int = 8007, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class NextTry(Exception):
        def __init__(self, error_message: str, error_code: int = 8008, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class MaxRetry(StopIteration):
        def __init__(self, error_message: str, error_code: int = 8009, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class CacheError(ValueError):
        def __init__(self, error_message: str, error_code: int = 8010, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])


Exceptions = CodedExceptions()
