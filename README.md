# py_erp
## A solution to non-existing problem

This Python-based GUI application provides tools for processing and visualizing Event-Related Potential (ERP) data. It allows users to:

1.  Convert ERPLAB `.erp` files to MNE-Python compatible `.fif` files.
2.  Load and plot ERP waveforms from `.fif` files with various customization options.

The application features a dark-themed interface for controls and a light-themed, publication-ready plot display.

---
## Features

* **ERPLAB to MNE Conversion**: Easily convert `.erp` files (MATLAB-based) into the MNE-Python native `.fif` format, making your data accessible for further analysis within the MNE ecosystem.
* **Interactive ERP Plotting**:
    * Load single or multiple evoked conditions from a `.fif` file.
    * Specify channels for plotting (e.g., `P8`, `Cz,Fz`).
    * Select specific bins (1-based indexing) to display (e.g., `1,2`, `4,5,6`).
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
