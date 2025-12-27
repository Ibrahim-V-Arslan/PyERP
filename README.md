***

# PyERP
## A solution to a non-existing problem

This Python-based GUI application provides tools for processing and visualizing Event-Related Potential (ERP) data specifically from **ERPLAB**/**EEGLAB**. It allows users to:

1.  Convert ERPLAB/EEGLAB `.erp/.set` files to MNE-Python compatible `.fif` files.
2.  Load and plot ERP waveforms from `.fif` files with various customization options.

The application features a dark-themed interface for controls and a light-themed, publication-ready plot display.

---

1.  Ensure you have all the required libraries listed under "Requirements" below.
2.  Save the code as a Python file (e.g., `py_erp.py`).
3.  Run it from your terminal:
    ```bash
    python py_erp.py
    ```
---

You can also compile this using Pyinstaller (Win) or Py2App (Mac-silicon) so you can have a standalone app.
You can reach out to me for the specific setup files if you have any trouble.
## Features

* **ERPLAB/EEGLAB to MNE Conversion**: Easily convert `.erp/.set` files (MATLAB-based) into the MNE-Python native `.fif` format, making your data accessible for further analysis within the MNE ecosystem.
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
    * Add vertical lines at specified time points (e.g., for stimulus onset, component peaks).
* **Plot Export**: Save the generated plot in various formats (PNG, JPEG, SVG, PDF) with adjustable DPI.
* **User-Friendly Interface**:
    * Dark theme for application controls for comfortable viewing.
    * Standard light theme for the plot area for clarity and easy integration into documents.
    * File dialogs for easy selection of input and output files.
    * A confirmation dialog with a quote prevents accidental closing of the application.
    * Informative messages and error handling.

---
## Requirements

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
