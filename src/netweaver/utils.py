import matplotlib

matplotlib.use("module://ipympl.backend_nbagg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from ._internal_utils import get_dirname, join_path


class PlotTraining:
    """
    Provides tools for visualizing training progress of machine learning models.

    This class supports both live and static plotting of training, validation, and batch logs, including loss, accuracy, and learning rate.
    """

    def __init__(self, *, path_epoch_log=None, path_validation_log=None, path_batch_log=None):
        """
        Initializes the PlotTraining class with optional log file paths.

        This constructor sets up the file paths for epoch, validation, and batch logs, and initializes the animation attribute.

        Parameters
        ----------
        path_epoch_log : str, optional
            Path to the epoch log CSV file.
        path_validation_log : str, optional
            Path to the validation log CSV file.
        path_batch_log : str, optional
            Path to the batch log CSV file.

        Returns
        -------
        None
        """
        self.path_epoch_log = path_epoch_log
        self.path_validation_log = path_validation_log
        self.path_batch_log = path_batch_log
        self.path_model_dir = get_dirname(self.path_epoch_log)
        self.ani = None

    def live_plot(self, *, xtp_slips=(30, 15), loss_caplimit=1.5, acc_caplimit=1.1, lr_caplimit=0.06):
        """
        Displays a live-updating plot of training, validation, and batch metrics.

        This method visualizes loss, accuracy, and learning rate in real time as the training progresses.

        Parameters
        ----------
        xtp_slips : tuple, optional
            Number of epochs to display xtp_slips[0] and the slip window for x-axis xtp_slips[1].
        loss_caplimit : float, optional
            Maximum value for the loss y-axis.
        acc_caplimit : float, optional
            Maximum value for the accuracy y-axis.
        lr_caplimit : float, optional
            Maximum value for the learning rate y-axis.

        Returns
        -------
        None
        """
        self.xtp_slips = xtp_slips

        self.last_read_row_batch = 0
        self.last_read_row_epoch = 0
        self.last_read_row_validation = 0

        self.fig, self.axes_loss = plt.subplots(nrows=1, ncols=1, figsize=(10, 7.5))
        self.axes_acc = self.axes_loss.twinx()
        self.axes_lr = self.axes_loss.twinx()

        self.axes_loss.set_xlabel("Epoch")
        self.axes_loss.set_ylabel("Loss")
        self.axes_acc.set_ylabel("Accuracy")
        self.axes_lr.set_ylabel("LR decay")
        self.axes_lr.spines["right"].set_position(("outward", 60))

        self.axes_loss.set_ylim(-0.1, loss_caplimit)
        self.axes_loss.set_xlim(0, self.xtp_slips[0])
        self.axes_acc.set_ylim(-0.1, acc_caplimit)
        self.axes_lr.set_ylim(-0.01, lr_caplimit)

        self.axes_loss.grid(True)
        self.axes_acc.grid(False)
        self.axes_lr.grid(False)

        self.x_epoch = []
        self.y_epoch_loss = []
        self.y_epoch_accuracy = []

        self.x_batch_epoch = []
        self.y_batch_lr = []

        self.x_val_epoch = []
        self.y_val_loss = []
        self.y_val_acc = []

        if self.path_epoch_log is not None:
            (self.artist_eploss,) = self.axes_loss.plot([], [], label="Epoch Loss", color="blue", linewidth=2)
            (self.artist_epacc,) = self.axes_acc.plot([], [], label="Epoch Acc", color="orange", linewidth=2)
            # (self.artist_lr,) = self.axes_lr.plot([], [], label="LR decay", color="green", alpha=0.4, linestyle="-", linewidth=2)

        if self.path_validation_log is not None:
            (self.artist_vloss,) = self.axes_loss.plot([], [], label="Val Loss", color="blue", alpha=0.5, linewidth=2)
            (self.artist_vacc,) = self.axes_acc.plot([], [], label="Val Acc", color="orange", alpha=0.5, linewidth=2)

        if self.path_batch_log is not None:
            (self.artist_lr,) = self.axes_lr.plot([], [], label="LR decay", color="grey", alpha=1, linestyle="-", linewidth=2)

        self.axes_loss.legend(loc="upper left")
        self.axes_acc.legend(loc="upper right")
        self.axes_lr.legend(loc="upper center")
        plt.tight_layout()

        self.ani = FuncAnimation(self.fig, self._animate, interval=500, blit=True)
        # return self.ani
        plt.show()

    def _animate(self, frame):
        """
        Updates the plot with new data for each animation frame.

        This method reads new rows from the log files and updates the plot artists accordingly.

        Parameters
        ----------
        frame : int
            The current animation frame (unused).

        Returns
        -------
        tuple
            A tuple of updated matplotlib artist objects.
        """
        try:
            return_list = []
            if self.path_epoch_log is not None:
                data_epoch = pd.read_csv(self.path_epoch_log, skiprows=range(1, self.last_read_row_epoch + 1))
                if not data_epoch.empty:
                    self.x_epoch.extend(data_epoch["epoch"].tolist())
                    self.y_epoch_loss.extend(data_epoch["loss_epoch"].tolist())
                    self.y_epoch_accuracy.extend(data_epoch["accuracy_epoch"].tolist())
                    # self.y_batch_lr.extend(data_epoch["learning_rate"].tolist())

                    self.artist_eploss.set_data(self.x_epoch, self.y_epoch_loss)
                    self.artist_epacc.set_data(self.x_epoch, self.y_epoch_accuracy)
                    # self.artist_lr.set_data(self.x_epoch, self.y_epoch_lr)

                    return_list.extend([self.artist_eploss, self.artist_epacc])

                    self.last_read_row_epoch += len(data_epoch)

            if self.path_batch_log is not None:  # skiprows is 0 index style. range(1, ...) excludes header
                data_batch = pd.read_csv(self.path_batch_log, skiprows=range(1, self.last_read_row_batch + 1))
                if not data_batch.empty:
                    self.x_batch_epoch.extend(data_batch["xaxis_value"].tolist())
                    self.y_batch_lr.extend(data_batch["learning_rate"].tolist())
                    self.artist_lr.set_data(self.x_batch_epoch, self.y_batch_lr)  # learning rate x axis value doesn't match with common x axis. idea?
                    # use ticks and subticks concept
                    return_list.extend([self.artist_lr])

                    self.last_read_row_batch += len(data_batch)  # len(pandas_data_file) doesn't include header in the count

            if self.path_validation_log is not None:
                data_validation = pd.read_csv(self.path_validation_log, skiprows=range(1, self.last_read_row_validation + 1))
                if not data_validation.empty:
                    self.x_val_epoch.extend(data_validation["epoch"].tolist())
                    self.y_val_loss.extend(data_validation["loss_validation"].tolist())
                    self.y_val_acc.extend(data_validation["accuracy_validation"].tolist())

                    self.artist_vloss.set_data(self.x_val_epoch, self.y_val_loss)
                    self.artist_vacc.set_data(self.x_val_epoch, self.y_val_acc)

                    return_list.extend([self.artist_vloss, self.artist_vacc])

                    self.last_read_row_validation += len(data_validation)

            current_xlim = self.axes_loss.get_xlim()
            if self.x_epoch[-1] > current_xlim[1] - 2:
                self.axes_loss.set_xlim(left=current_xlim[0] + self.xtp_slips[1], right=current_xlim[1] + self.xtp_slips[1])

            return tuple(return_list)

        except Exception as e:
            print(f"Error in _animate: {e}")

    def static_plot(self, save_to_file: bool = True):
        """
        Generates and displays a static plot of training, validation, and batch metrics.

        This method visualizes loss, accuracy, and learning rate from log files and optionally saves the plot as an image file.

        Parameters
        ----------
        save_to_file : bool, optional
            Whether to save the generated plot to a file. Defaults to True.

        Returns
        -------
        None
        """
        fig, axes_loss = plt.subplots(nrows=1, ncols=1, figsize=(10, 7.5))
        axes_acc = axes_loss.twinx()
        axes_lr = axes_loss.twinx()

        axes_lr.spines["right"].set_position(("outward", 60))
        axes_loss.set_xlabel("Epoch")
        axes_loss.set_ylabel("Loss")
        axes_acc.set_ylabel("Accuracy")
        axes_lr.set_ylabel("LR decay")

        # self.axes_loss.set_ylim(-0.1, loss_caplimit)
        # self.axes_loss.set_xlim(0, 30)
        # self.axes_acc.set_ylim(-0.1, 1.1)
        # self.axes_lr.set_ylim(-0.1, 1.1)

        axes_loss.grid(True)
        axes_acc.grid(False)
        axes_lr.grid(False)

        if self.path_epoch_log is not None:
            data_epoch = pd.read_csv(self.path_epoch_log)
            if not data_epoch.empty:
                (artist_eploss,) = axes_loss.plot(data_epoch["epoch"], data_epoch["loss_epoch"], label="Epoch Loss", color="blue", linewidth=2)
                (artist_epacc,) = axes_acc.plot(data_epoch["epoch"], data_epoch["accuracy_epoch"], label="Epoch Acc", color="orange", linewidth=2)
                # (artist_lr,) = axes_lr.plot(
                #     data_epoch["epoch"], data_epoch["learning_rate"], label="LR decay", color="grey", alpha=0.4, linestyle=":", linewidth=1
                # )

        if self.path_batch_log is not None:
            data_batch = pd.read_csv(self.path_batch_log)
            if not data_batch.empty:
                (artist_lr,) = axes_lr.plot(
                    data_batch["xaxis_value"], data_batch["learning_rate"], label="LR decay", color="grey", alpha=1, linestyle="-", linewidth=2
                )

        if self.path_validation_log is not None:
            data_validation = pd.read_csv(self.path_validation_log)
            if not data_validation.empty:
                (artist_vloss,) = axes_loss.plot(
                    data_validation["epoch"], data_validation["loss_validation"], label="Val Loss", color="blue", alpha=0.5, linewidth=2
                )
                (artist_vacc,) = axes_acc.plot(
                    data_validation["epoch"], data_validation["accuracy_validation"], label="Val Acc", color="orange", alpha=0.5, linewidth=2
                )

        axes_loss.legend(loc="upper left")
        axes_acc.legend(loc="upper right")
        axes_lr.legend(loc="upper center")

        # if self.path_batch_log is not None:
        #     (self.artist_lr,) = self.axes_lr.plot([], [], label="LR decay", color="grey", alpha=0.2)
        plt.tight_layout()
        if save_to_file:
            file_name = "training_graph.png"
            path_image_file = join_path(self.path_model_dir, file_name)
            plt.savefig(path_image_file)
        plt.show()

    def pause(self):
        """
        Pauses the live animation if it is currently running.

        This method temporarily halts the live plot updates until resumed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.ani is not None:
            self.ani.pause()

    def resume(self):
        """
        Resumes the live animation if it is currently paused.

        This method restarts the live plot updates after being paused.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.ani is not None:
            self.ani.event_source.start()
