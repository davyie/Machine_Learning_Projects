import matplotlib.pyplot as plt 
import seaborn as sns

class Visualizer():
    def __init__(self) -> None:
        pass
    
    def visualize_plot(self, feature_name, data, labels):
        chart = sns.countplot(x=feature_name, data=data)
        chart.set_xticklabels(labels=labels, rotation=90)
        plt.show()

