import tkinter as tk
from tkinter import scrolledtext
import logging # Using standard logging for file output << explained in README.md

class LogHandler:
    """
    A simple logger class that can direct messages to a Tkinter ScrolledText widget
    and optionally to a file using the standard `logging` module.
    """
    def __init__(self):
        self.text_widget = None
        self._file_logger = None
        self._gui_logging_enabled = True # Control if messages go to GUI

    def set_text_widget(self, widget: scrolledtext.ScrolledText):
        """Sets the Tkinter ScrolledText widget where messages should be displayed."""
        self.text_widget = widget
        self._gui_logging_enabled = True # Re-enable GUI logging if widget is set

    def enable_gui_logging(self):
        """Enables logging messages to the GUI text widget."""
        self._gui_logging_enabled = True

    def disable_gui_logging(self):
        """Disables logging messages to the GUI text widget."""
        self._gui_logging_enabled = False

    def set_file_logger(self, filepath: str, level=logging.INFO):
        """
        Sets up a file logger to write messages to a specified file.
        Clears existing file handlers to prevent duplicate logs.
        """
        self._file_logger = logging.getLogger('sim_file_logger')
        self._file_logger.setLevel(level)

        # Remove existing file handlers to prevent duplicate logging if called multiple times
        for handler in list(self._file_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                self._file_logger.removeHandler(handler)
                handler.close() # Important to close the file stream

        file_handler = logging.FileHandler(filepath)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self._file_logger.addHandler(file_handler)
        self._file_logger.propagate = False # Prevent messages from going to root logger

    def log(self, message: str, level='info'):
        """
        Logs a message to the console, Tkinter widget (if enabled), and/or file logger (if set).

        Args:
            message (str): The message string to log.
            level (str): The logging level ('info', 'warning', 'error', 'debug').
        """
        print(message) # Always print to console for immediate feedback

        if self._gui_logging_enabled and self.text_widget:
            self.text_widget.insert(tk.END, message + "\n")
            self.text_widget.see(tk.END)  # Scroll to the end
            self.text_widget.update_idletasks()  # Ensure GUI updates immediately

        if self._file_logger:
            if level == 'info':
                self._file_logger.info(message)
            elif level == 'warning':
                self._file_logger.warning(message)
            elif level == 'error':
                self._file_logger.error(message)
            elif level == 'debug':
                self._file_logger.debug(message)
            else: # Default to info for unrecognized levels
                self._file_logger.info(message)