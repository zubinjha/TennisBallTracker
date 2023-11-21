class ChartDataLogger:
    def __init__(self):
        self.all_positions = []  # Store all (x, y, r) tuples for charting

    def log_position(self, x, y, r):
        self.all_positions.append((x, y, r))

    def get_all_positions(self):
        return self.all_positions
    
    def get_positions_local_avg(self, last_n_frames):
        # This will store the local averages
        local_averages = []

        # Iterate through all_positions to calculate local averages
        for i in range(len(self.all_positions)):
            # Determine the start index for averaging
            start_index = max(i - last_n_frames + 1, 0)
            end_index = i + 1  # End index is exclusive

            # Slice the portion of the list to average
            subset = self.all_positions[start_index:end_index]

            # Calculate the average for each component (x, y, r)
            avg_x = sum(pos[0] for pos in subset) / len(subset)
            avg_y = sum(pos[1] for pos in subset) / len(subset)
            avg_r = sum(pos[2] for pos in subset) / len(subset)

            # Append this average to the list
            local_averages.append((avg_x, avg_y, avg_r))

        return local_averages