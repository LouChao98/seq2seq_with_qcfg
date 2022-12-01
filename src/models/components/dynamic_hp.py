import numpy as np


class DynamicHyperParameter:
    def __init__(self, control_points) -> None:
        if isinstance(control_points, (int, float)):
            self.values = [control_points]
        else:
            # control_points: [value@epoch], value is evaluated to float
            parsed_control_points = []
            for item in control_points:
                v, e = item.split("@")
                v = float(v)
                e = int(e)
                parsed_control_points.append((v, e))
            if parsed_control_points[0][1] != 0:
                parsed_control_points.insert(0, (parsed_control_points[0][0], 0))
            self.values = []
            for i in range(len(parsed_control_points) - 1):
                start = parsed_control_points[i]
                end = parsed_control_points[i + 1]
                step_size = (end[0] - start[0]) / (end[1] - start[1])
                for j, _ in enumerate(range(start[1], end[1])):
                    self.values.append(start[0] + step_size * j)
            self.values.append(end[0])

    def get(self, epoch):
        epoch = min(epoch, len(self.values) - 1)
        return self.values[epoch]
