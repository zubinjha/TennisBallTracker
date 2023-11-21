class RecentFrameTracker:
    def __init__(self, numFrames):
        self.numFrames = numFrames
        self.positions = []  # Store recent (x, y, r) tuples
        self.avg_positions = []  # Store the average positions

    def log_position(self, x, y, r):
        self.positions.append((x, y, r))
        if len(self.positions) > self.numFrames:
            self.positions.pop(0)
        self.update_avg_position()

    def update_avg_position(self):
        if self.positions:
            avg_x = sum(p[0] for p in self.positions) / len(self.positions)
            avg_y = sum(p[1] for p in self.positions) / len(self.positions)
            avg_r = sum(p[2] for p in self.positions) / len(self.positions)
            self.avg_positions.append((avg_x, avg_y, avg_r))

    def get_avg(self):
        if not self.positions:
            return None
        avg_x = sum(p[0] for p in self.positions) / len(self.positions)
        avg_y = sum(p[1] for p in self.positions) / len(self.positions)
        avg_r = sum(p[2] for p in self.positions) / len(self.positions)
        return (avg_x, avg_y, avg_r)