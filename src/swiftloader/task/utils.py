


class AnyDict():
    def __init__(self, value):
        self.value = value
        
    def __eq__(self, other):
        return self.value == other.value
    
    def __ne__(self, other):
        return self.value != other.value
    
    def __repr__(self):
        return self.value.__repr__()
    
    def __str__(self):
        return self.value.__str__()
    
    def __getitem__(self, key):
        return self.value