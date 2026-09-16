class SideHint:
    LEFT = "left"
    RIGHT = "right"
    NUMBER = "number"
    COLOR_NAME = "fieldPlayerColor"

    def __init__(self, side, hint_str):
        if side not in {self.LEFT, self.RIGHT}:
            raise ValueError('Only "left" and "right" are permitted as sides.')
        self.side = side

        try:
            self.hint = int(hint_str)
            self.hint_type = self.NUMBER
        except ValueError:  # not an int
            self.hint = hint_str
            self.hint_type = self.COLOR_NAME

    @staticmethod
    def get_serialization(side, hint):
        return f"{side}, {hint}"
    def dumps(self):
        return self.get_serialization(self.side, self.hint)
    
    @classmethod
    def loads(cls, string):
        return cls(*[s.strip() for s in string.split(",")])
