import os
import pickle
from datetime import datetime


class DataLogger:
    """
    A simple data logger that collects data entries and saves them to a pickle file.
    """

    def __init__(self, log_dir="logs"):
        """
        Initializes the logger.

        Args:
            log_dir (str): The directory where log files will be stored. If None, logging is disabled.
        """
        self.log_data = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(log_dir, f"teleop_log_{timestamp}.pkl")
        else:
            self.log_file = None

    def add_entry(self, data_entry):
        """
        Adds a data entry to the log. A timestamp is automatically added.

        Args:
            data_entry (dict): A dictionary containing the data to log for the current timestep.
        """
        if not self.log_file:
            return
        self.log_data.append(data_entry)

    def save(self):
        """
        Saves the collected log data to a pickle file.
        """
        if not self.log_file or not self.log_data:
            print("No data to save or logging is disabled.")
            return

        print(f"Saving {len(self.log_data)} data points to {self.log_file}...")
        try:
            with open(self.log_file, "wb") as f:
                pickle.dump(self.log_data, f)
            print(f"Data successfully saved to {self.log_file}")
        except IOError as e:
            print(f"Error saving data: {e}")
