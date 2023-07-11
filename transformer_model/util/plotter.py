import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import clear_output

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        
        if not hasattr(self, "metrics"): 
            self.metrics = {}
            self.epoch = 0
            for metric in logs:
                self.metrics[metric] = []
            plt.ion()
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        if self.epoch == 0:
            self.f, self.axs = plt.subplots(1, len(metrics), figsize=(15,5))
        f = self.f
        axs = self.axs

        clear_output(wait=True)
        

        for i, metric in enumerate(metrics):
            axs[i].cla()
            axs[i].plot(range(1, self.epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, self.epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()
        
        self.epoch += 1

        plt.tight_layout()
        f.canvas.draw()
        f.canvas.flush_events()
    

    def finish(self, save_name=None):
        if save_name:
            plt.savefig(f'./plots/{save_name}.png')
        plt.ioff()
        plt.show()
