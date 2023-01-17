from __future__ import annotations

__all__ = ["AbstractTarget"]


from typing import Any, Mapping


class AbstractTarget:
    def validate_input(self, value: Any) -> Any:
        return value

    def preprocess_input(self, value: Any) -> Any:
        return value

    def postprocess_result(self, value: Any) -> Any:
        return value

    def validate_targets(self, targets: Mapping[str, AbstractTarget]):
        """
        Class ansectors can use this method to validate present targets w.r.t to requirements of the given target.
        For instance, presence or absence of a certain type of targets.
        If something is wrong, this method should raise an ConfigurationException.

        Args:
            targets: All existing targets in the pipeline (including the current one)

        Returns:
            None
        """

        pass
