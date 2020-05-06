class Segment:
    def __init__(self, segment, label):
        self.segment = segment / 255.0
        self.label = label
        # self.center = self.get_center()

    # def get_center(self):
    #     center = int(len(self.target) // 2)
    #     return int(self.target[center, center] / 255)

