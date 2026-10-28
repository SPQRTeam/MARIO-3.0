# this looks different b/c the base is copied from "codename lucia-wm"

import dataclasses
from functools import cached_property
import yaml

def scalable_field():
    return dataclasses.field(metadata={"scale": True})

@dataclasses.dataclass
class FieldConfig(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!field"

    width: float = scalable_field()
    height: float = scalable_field()
    center_radius: float = scalable_field()
    goal_area_width: float = scalable_field()
    goal_area_height: float = scalable_field()
    penalty_area_width: float = scalable_field()
    penalty_area_height: float = scalable_field()
    penalty_mark_distance: float = scalable_field()
    goal_width: float = scalable_field()
    goal_height: float = scalable_field()
    goal_depth: float = scalable_field()
    line_width: float = scalable_field()
    ball_radius: float = scalable_field()

    # just for forcing standard __init__ (for the palette)
    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return cls(**mapping)

    @cached_property
    def xlimit(self):
        return self.width / 2

    @cached_property
    def ylimit(self):
        return self.height / 2

    @cached_property
    def goal_area_ylimit(self):
        return self.goal_area_height / 2

    @cached_property
    def penalty_area_ylimit(self):
        return self.penalty_area_height / 2

    @cached_property
    def penalty_mark_x(self):
        return self.xlimit - self.penalty_mark_distance

    @cached_property
    def goalpost_y(self):
        # note: goal_width is correct below, goal measures have a different nomenclature than field area measures
        return self.goal_width / 2

    # for example, convert meters to millimeters
    def scale(self, scalar):
        for f in dataclasses.fields(self):
            if f.metadata.get("scale", False):
                current_value = getattr(self, f.name)
                setattr(self, f.name, current_value * scalar)
        self._clear_cache()

    # Finds all cached_properties on the class and clears them from the instance
    def _clear_cache(self):
        for key, attr in type(self).__dict__.items():
            if isinstance(attr, cached_property):
                self.__dict__.pop(key, None)