class Segment:
    def __init__(self, segment, target):
        self.segment = segment
        self.target = target
        self.center = self.get_center()

    def get_center(self):
        center = int(len(self.target) // 2)
        return int(self.target[center, center] / 255)
