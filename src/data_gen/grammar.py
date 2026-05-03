
class Breaker:
    def __init__(self, cls, width):
        self.cls = cls
        self.width = width # in modules

class Rail:
    def __init__(self, max_modules=12):
        self.max_modules = max_modules
        self.components = []
        self._total_width = 0 

    def add_component(self, breaker):
        if (self._total_width + breaker.width) <= self.max_modules:
            self._total_width += breaker.width
            self.components.append(breaker)
            return True 
        else:
            return False
    def current_width(self):
        return self._total_width

class Panel:
    def __init__(self, rails_count=2, rail_height=200):
        self.rail_height = rail_height
        self.rails = [Rail() for _ in range (rails_count)]
    def get_rail_y_center(self, rail_index):
        half_height = self.rail_height / 2
        return (rail_index * self.rail_height) + half_height