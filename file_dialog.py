import tkinter as tk
from tkinter import filedialog
import os


def open_file_dialog(title="Open File", initial_dir=None, file_types=None):
    """
    Display a file open dialog and return the selected file path.

    Args:
        title (str): Title of the dialog window
        initial_dir (str): Initial directory to open the dialog in (defaults to current directory)
        file_types (list): List of tuples with descriptions and file extensions, e.g. [("Video files", "*.mp4;*.avi")]

    Returns:
        str: The selected file path, or None if canceled
    """
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Set the dialog to be on top of other windows
    root.attributes("-topmost", True)

    # Use initial_dir if provided, otherwise use the current directory
    if initial_dir is None:
        initial_dir = os.getcwd()

    # Set default file_types if not provided
    if file_types is None:
        file_types = [
            ("All files", "*.*"),
            ("Video files", "*.mp4;*.mov;*.avi"),
            ("Image files", "*.jpg;*.jpeg;*.png"),
        ]

    # Show the file dialog
    file_path = filedialog.askopenfilename(
        title=title, initialdir=initial_dir, filetypes=file_types
    )

    # Destroy the root window
    root.destroy()

    # Return the selected file path (or empty string if canceled)
    return file_path if file_path else None


def save_file_dialog(
    title="Save File", initial_dir=None, default_extension=None, file_types=None
):
    """
    Display a file save dialog and return the selected file path.

    Args:
        title (str): Title of the dialog window
        initial_dir (str): Initial directory to open the dialog in (defaults to current directory)
        default_extension (str): Default extension to add if none is specified
        file_types (list): List of tuples with descriptions and file extensions, e.g. [("Text files", "*.txt")]

    Returns:
        str: The selected file path, or None if canceled
    """
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Set the dialog to be on top of other windows
    root.attributes("-topmost", True)

    # Use initial_dir if provided, otherwise use the current directory
    if initial_dir is None:
        initial_dir = os.getcwd()

    # Set default file_types if not provided
    if file_types is None:
        file_types = [
            ("All files", "*.*"),
            ("Text files", "*.txt"),
            ("JSON files", "*.json"),
        ]

    # Show the file dialog
    file_path = filedialog.asksaveasfilename(
        title=title,
        initialdir=initial_dir,
        defaultextension=default_extension,
        filetypes=file_types,
    )

    # Destroy the root window
    root.destroy()

    # Return the selected file path (or empty string if canceled)
    return file_path if file_path else None
