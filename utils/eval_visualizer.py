from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import cm


class EvalVisualizer():
    def __init__(self, data) -> None:
        self.data = data
        self.run()

    def run(self):
        self.all_xai_methods = list(self.data.keys())
        self.all_point_generators = list(
            self.data[list(self.data.keys())[0]].keys())
        self.all_evals = list(self.data[list(self.data.keys())[0]][list(
            self.data[list(self.data.keys())[0]].keys())[0]].keys())
        self.visualize()

    def visualize(self):
        max_val = 0
        max_data = ""
        # Loop over all evaluation methods and create a subgraph for each
        fig, axs = plt.subplots(len(self.all_evals), 1,
                                figsize=(20 * len(self.all_evals), 5))
        if len(self.all_evals) == 1:
            axs = [axs]
        for idx, eval_method in enumerate(self.all_evals):
            # Create a grouped bar chart where the x-axis is the XAI methods,
            # and the y-axis is the average of the evaluation method
            x = np.arange(len(self.all_xai_methods))
            bar_width = 0.08
            colors = cm.rainbow(np.linspace(
                0, 1, len(self.all_point_generators)))
            for pg_idx, point_generator in enumerate(self.all_point_generators):
                y = []
                for xai_method in self.all_xai_methods:
                    y.append(self.average_entry(
                        self.data[xai_method][point_generator][eval_method]))
                    if y[-1] > max_val:
                        max_val = y[-1]
                        max_data = f"{xai_method} {point_generator} {eval_method} {y[-1]}"

                # Offset by bar_width for each point generator
                axs[idx].bar(x + pg_idx * bar_width, y,
                             bar_width, label=point_generator, color=colors[pg_idx])
            axs[idx].set_xticks(
                x + bar_width * (len(self.all_point_generators) - 1) / 2)
            axs[idx].set_xticklabels(self.all_xai_methods)
            axs[idx].set_title(eval_method)
            # Place the legend outside the graph
            axs[idx].legend(title='Point Generators',
                            bbox_to_anchor=(1.05, 1), loc='upper left')
        print(max_data)
        plt.tight_layout()
        plt.show()

    def average_entry(self, entry):
        l = len(entry)
        # entry = list(map(lambda x: max(x), entry))
        entry = list(map(lambda x: x[0], entry))
        avrg = sum(entry) / l
        return avrg


if __name__ == "__main__":
    import json
    with open("back_prop_all100.json") as f:
        data = json.load(f)
    ev = EvalVisualizer(data)
