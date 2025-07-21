from abc import ABC, abstractmethod


class BaseCameraInterface(ABC):
    """
    An abstract base class for camera interfaces.
    """

    @abstractmethod
    def start(self):
        """Starts the camera stream."""
        pass

    @abstractmethod
    def stop(self):
        """Stops the camera stream."""
        pass

    @abstractmethod
    def update_frames(self):
        """
        Updates the frames from the camera source.
        For polling-based interfaces, this should fetch new frames.
        For callback-based interfaces, this might be a no-op.
        """
        pass

    @abstractmethod
    def get_frames(self):
        """
        Fetches and returns frames from all cameras.

        Returns:
            dict: A dictionary where keys are camera identifiers (e.g., serial numbers)
                  and values are another dictionary containing 'color' and 'depth' numpy arrays.
        """
        pass

    @abstractmethod
    def get_frame(self, identifier: str):
        """
        Fetches frames from a specific camera.

        Args:
            identifier (str): The identifier of the camera.

        Returns:
            dict: A dictionary containing 'color' and 'depth' numpy arrays.
        """
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
