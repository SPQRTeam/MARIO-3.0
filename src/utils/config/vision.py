from munch import Munch

class VisionConfig(Munch):
    def __init__(self, data):
        super().__init__(data)

    def equivalent_colors(self, color):
        color = color.lower()
        try:
            return set(self._equivalent_colors[color]) | {color}
        except KeyError:  # color not in self._equivalent_colors (don't want the user to tediously write every possible color in the config)
            return {color}
