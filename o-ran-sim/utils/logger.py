# utils/logger.py

import tkinter as tk
from tkinter import scrolledtext

class LogHandler:
    """
    A simple logger class that can direct messages to a Tkinter ScrolledText widget.
    """
    def __init__(self):
        self.text_widget = None

    def set_text_widget(self, widget: scrolledtext.ScrolledText):
        """Sets the Tkinter ScrolledText widget where messages should be displayed."""
        self.text_widget = widget

    def log(self, message: str):
        """Logs a message to the console and/or the Tkinter widget."""
        print(message) # Always print to console for debugging/non-GUI use
        if self.text_widget:
            self.text_widget.insert(tk.END, message + "\n")
            self.text_widget.see(tk.END) # Scroll to the end
            self.text_widget.update_idletasks() # Ensure GUI updates immediately