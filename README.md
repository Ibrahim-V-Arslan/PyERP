***

# PyERP
## A solution to a non-existing problem

This Python-based GUI application provides tools for processing and visualizing Event-Related Potential (ERP) data specifically from **ERPLAB**. It allows users to:

1.  Convert ERPLAB `.erp` files to MNE-Python compatible `.fif` files.
2.  Load and plot ERP waveforms from `.fif` files with various customization options.

The application features a dark-themed interface for controls and a light-themed, publication-ready plot display.

---
## Installation

You have two options to run this application:

### Standalone Executable (Recommended)
This is the easiest method and does not require a Python installation. (Compiled with Pyinstaller)

1.  Download the `pyerp.exe` file. ([Download Link](https://drive.google.com/file/d/1RtZdYQG_SjQvH0U0nm4GPQYMvJ46BRog/view?usp=sharing))
2.  Double-click the executable to launch the application.

### Running from Source
This method is for users who have Python installed and want to run the code directly.

1.  Ensure you have all the required libraries listed under "Requirements" below.
2.  Save the code as a Python file (e.g., `py_erp.py`).
3.  Run it from your terminal:
    ```bash
    python py_erp.py
    ```
---
## Features

* **ERPLAB to MNE Conversion**: Easily convert `.erp` files (MATLAB-based) into the MNE-Python native `.fif` format, making your data accessible for further analysis within the MNE ecosystem.
* **Customizable Legend Editing**: Interactively edit the labels of plotted waveforms after they are generated, allowing for clear and publication-ready figure legends.
* **Interactive ERP Plotting**:
    * Load single or multiple evoked conditions from a `.fif` file.
    * Specify channels for plotting by typing them or by selecting from a list in an interactive browser window.
    * Select specific bins (1-based indexing) to display by typing them or choosing from an interactive browser that shows bin names/comments.
    * Optionally average data across selected channels.
    * Optionally average data across selected bins.
    * Customize the time window (start and end in milliseconds).
    * Set custom Y-axis limits (in microvolts).
    * Add a custom plot title.
    * Toggle gridlines on the plot.
    * Add vertical lines at specified time points (e.g., for stimulus onset, component peaks).
* **Plot Export**: Save the generated plot in various formats (PNG, JPEG, SVG, PDF) with adjustable DPI.
* **User-Friendly Interface**:
    * Dark theme for application controls for comfortable viewing.
    * Standard light theme for the plot area for clarity and easy integration into documents.
    * File dialogs for easy selection of input and output files.
    * A confirmation dialog with a quote prevents accidental closing of the application.
    * Informative messages and error handling.

---
## Requirements (for Running from Source)

* Python 3.x
* **tkinter**: Usually included with standard Python installations.
* **scipy**: For loading `.erp` (MATLAB) files.
* **numpy**: For numerical operations.
* **mne**: For ERP data handling and `.fif` file operations.
* **matplotlib**: For plotting.

You can install the necessary libraries using pip:
```bash
pip install scipy numpy mne matplotlib
```
