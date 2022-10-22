__all__ = ["AbstractTarget"]

from typing import Any


class AbstractTarget:
    def validate_input(self, value: Any) -> Any:
        return value

    def preprocess_input(self, value: Any) -> Any:
        return value

    def postprocess_result(self, value: Any) -> Any:
        return value
