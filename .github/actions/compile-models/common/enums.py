"""
Enums declaration
"""

from enum import Enum


class Status(Enum):
    """Enum class defining possible compilation statuses"""

    SUCCESS = 1
    SKIPPED = 2
    DISABLED = 3
    FAILED = 4
    TIMEOUT = 5
    MODEL_NOT_FOUND = 6

    @classmethod
    def get_ok_statuses(cls):
        """Returns a set of statuses that don't throw an error"""
        return {cls.SUCCESS, cls.SKIPPED, cls.DISABLED}

    @classmethod
    def get_error_statuses(cls):
        """Returns a set of statuses that causes the script to fail"""
        return {cls.FAILED, cls.TIMEOUT, cls.MODEL_NOT_FOUND}
