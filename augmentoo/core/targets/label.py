from typing import Union, Mapping

from augmentoo.core.exceptions import ConfigurationException
from augmentoo.core.targets.abstract_target import AbstractTarget


class LabelTarget(AbstractTarget):
    """ """

    associated_target: Union[str, None] = None

    def validate_targets(self, targets: Mapping[str, AbstractTarget]):
        if self.associated_target is not None:
            if self.associated_target not in targets:
                raise ConfigurationException(
                    f"Associated target {self.associated_target} is not present in the pipeline"
                )
