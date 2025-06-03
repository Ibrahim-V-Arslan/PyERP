import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import scipy.io
import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Dark Mode Color Palette (for GUI elements, not the plot) ---
DARK_BG = "#2E2E2E"
DARK_FG = "#EAEAEA"
DARK_ENTRY_BG = "#3C3C3C"
DARK_ENTRY_FG = DARK_FG
DARK_BUTTON_BG = "#555555"
DARK_BUTTON_FG = DARK_FG
DARK_ACTIVE_BUTTON_BG = "#6A6A6A"
DARK_FRAME_BG = DARK_BG
DARK_FRAME_FG = "#CCCCCC" # For LabelFrame titles (for dark frames)
DARK_SELECT_COLOR = "#4A4A4A"
DARK_INSERT_BG = DARK_FG
READONLY_ENTRY_BG = "#333333"

# --- Plot Specific Colors (for light theme plot) ---
DEFAULT_PLOT_ZERO_LINE_COLOR = 'k' # Black for light background
USER_VLINE_COLOR = "red" # Red is visible on light backgrounds too

# --- Core MNE Processing Functions ---
def erp_to_fif(input_file, output_file):
    """
    Convert an ERPLAB .erp file to a single MNE-compatible .fif file.
    (Function remains unchanged)
    """
    try:
        erp_data = scipy.io.loadmat(input_file)
        erp = erp_data['ERP']
        bindata = erp['bindata'][0][0]
        times_ms = erp['times'][0][0][0]
        srate = erp['srate'][0][0][0][0]
        chanlocs = erp['chanlocs'][0][0]
        nbin = erp['nbin'][0][0][0][0]
        times_s = times_ms / 1000.0

        if 'labels' in chanlocs.dtype.names and chanlocs['labels'][0].size > 0:
            ch_names = [str(ch[0]) for ch in chanlocs['labels'][0]]
        else:
            num_channels = bindata.shape[0]
            ch_names = [f'EEG{i+1:03}' for i in range(num_channels)]
            print(f"Warning: Channel labels not found. Using default names: {ch_names[:3]}...")

        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=['eeg'] * len(ch_names))
        if info['sfreq'] <= 0:
            raise ValueError("Sampling frequency must be positive.")

        evoked_list = []
        for bin_idx in range(nbin):
            bin_data_volts = bindata[:, :, bin_idx] * 1e-6
            evoked = mne.EvokedArray(bin_data_volts, info, tmin=times_s[0], comment=f"Bin {bin_idx + 1}")
            evoked_list.append(evoked)

        mne.write_evokeds(output_file, evoked_list, overwrite=True)
        return True, f"Successfully converted '{os.path.basename(input_file)}' to '{os.path.basename(output_file)}'"
    except Exception as e:
        error_message = f"An error occurred during ERP to FIF conversion: {e}\n"
        error_message += f"Input file: {input_file}\n"
        if 'erp_data' in locals() and 'ERP' in erp_data:
            error_message += f"ERP structure keys: {erp_data['ERP'].dtype.names if hasattr(erp_data['ERP'], 'dtype') else 'N/A'}\n"
        return False, error_message

# --- Tkinter Application Class ---
class ErpProcessorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("ERP/FIF Processor")
        self.root.configure(bg=DARK_BG) # Main window background is dark
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Matplotlib will use its default (light) style. No dark style applied here.

        # --- Variables ---
        self.current_fif_file = tk.StringVar()
        self.evokeds_data = None
        self.avg_channels_var = tk.BooleanVar(value=False)
        self.avg_bins_var = tk.BooleanVar(value=False)
        self.gridlines_var = tk.BooleanVar(value=True)

        # --- GUI Frames ---
        # Dark themed frames for controls
        convert_frame = tk.LabelFrame(self.root, text="ERP to FIF Conversion", padx=10, pady=10, bg=DARK_FRAME_BG, fg=DARK_FRAME_FG, relief=tk.GROOVE)
        convert_frame.pack(fill=tk.X, padx=10, pady=5)

        plot_controls_frame = tk.LabelFrame(self.root, text="Plotting Controls", padx=10, pady=10, bg=DARK_FRAME_BG, fg=DARK_FRAME_FG, relief=tk.GROOVE)
        plot_controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Plot display frame will be default (light) themed
        self.plot_display_frame = tk.LabelFrame(self.root, text="Plot Display", padx=10, pady=10, relief=tk.GROOVE)
        # No bg/fg set, so it uses Tkinter default system colors.
        self.plot_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- Conversion Section Widgets (Dark Theme) ---
        tk.Label(convert_frame, text="Input .erp file:", bg=DARK_BG, fg=DARK_FG).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_erp_entry = tk.Entry(convert_frame, width=60, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.input_erp_entry.grid(row=0, column=1, sticky=tk.EW, pady=2)
        tk.Button(convert_frame, text="Browse ERP", command=self._browse_input_erp, bg=DARK_BUTTON_BG, fg=DARK_BUTTON_FG, activebackground=DARK_ACTIVE_BUTTON_BG, activeforeground=DARK_FG, relief=tk.FLAT, padx=5).grid(row=0, column=2, padx=5, pady=2)

        tk.Label(convert_frame, text="Output .fif file:", bg=DARK_BG, fg=DARK_FG).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_fif_entry = tk.Entry(convert_frame, width=60, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.output_fif_entry.grid(row=1, column=1, sticky=tk.EW, pady=2)
        tk.Button(convert_frame, text="Browse FIF Output", command=self._browse_output_fif, bg=DARK_BUTTON_BG, fg=DARK_BUTTON_FG, activebackground=DARK_ACTIVE_BUTTON_BG, activeforeground=DARK_FG, relief=tk.FLAT, padx=5).grid(row=1, column=2, padx=5, pady=2)
        
        tk.Button(convert_frame, text="Convert ERP to FIF", command=self._run_conversion, bg=DARK_BUTTON_BG, fg=DARK_BUTTON_FG, activebackground=DARK_ACTIVE_BUTTON_BG, activeforeground=DARK_FG, relief=tk.FLAT, padx=5).grid(row=2, column=1, pady=10)
        convert_frame.columnconfigure(1, weight=1)

        # --- Plotting Controls Section Widgets (Dark Theme) ---
        tk.Label(plot_controls_frame, text="Load .fif file:", bg=DARK_BG, fg=DARK_FG).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.load_fif_entry = tk.Entry(plot_controls_frame, textvariable=self.current_fif_file, width=50, state='readonly', readonlybackground=READONLY_ENTRY_BG, fg=DARK_ENTRY_FG, relief=tk.FLAT)
        self.load_fif_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW, pady=2)
        tk.Button(plot_controls_frame, text="Browse and Load FIF", command=self._browse_and_load_fif, bg=DARK_BUTTON_BG, fg=DARK_BUTTON_FG, activebackground=DARK_ACTIVE_BUTTON_BG, activeforeground=DARK_FG, relief=tk.FLAT, padx=5).grid(row=0, column=4, padx=5, pady=2)

        tk.Label(plot_controls_frame, text="Channel(s) (comma-sep):", bg=DARK_BG, fg=DARK_FG).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.channels_entry = tk.Entry(plot_controls_frame, width=25, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.channels_entry.insert(0, "P8")
        self.channels_entry.grid(row=1, column=1, sticky=tk.W, pady=2)

        tk.Label(plot_controls_frame, text="Bin Indices (comma-sep, 1-based):", bg=DARK_BG, fg=DARK_FG).grid(row=1, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.bins_entry = tk.Entry(plot_controls_frame, width=25, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.bins_entry.insert(0, "4,5")
        self.bins_entry.grid(row=1, column=3, columnspan=2, sticky=tk.W, pady=2)
        
        tk.Checkbutton(plot_controls_frame, text="Average Channels", variable=self.avg_channels_var, bg=DARK_BG, fg=DARK_FG, selectcolor=DARK_SELECT_COLOR, activebackground=DARK_BG, activeforeground=DARK_FG, highlightthickness=0).grid(row=2, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(plot_controls_frame, text="Average Bins", variable=self.avg_bins_var, bg=DARK_BG, fg=DARK_FG, selectcolor=DARK_SELECT_COLOR, activebackground=DARK_BG, activeforeground=DARK_FG, highlightthickness=0).grid(row=2, column=2, sticky=tk.W, padx=(10,0), pady=2)
        tk.Checkbutton(plot_controls_frame, text="Gridlines", variable=self.gridlines_var, bg=DARK_BG, fg=DARK_FG, selectcolor=DARK_SELECT_COLOR, activebackground=DARK_BG, activeforeground=DARK_FG, highlightthickness=0).grid(row=2, column=3, columnspan=2, sticky=tk.W, padx=(10,0), pady=2)

        tk.Label(plot_controls_frame, text="Time Start (ms):", bg=DARK_BG, fg=DARK_FG).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.time_start_entry = tk.Entry(plot_controls_frame, width=10, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.time_start_entry.insert(0, "0")
        self.time_start_entry.grid(row=3, column=1, sticky=tk.W, pady=2)

        tk.Label(plot_controls_frame, text="Time End (ms):", bg=DARK_BG, fg=DARK_FG).grid(row=3, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.time_end_entry = tk.Entry(plot_controls_frame, width=10, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.time_end_entry.insert(0, "600")
        self.time_end_entry.grid(row=3, column=3, columnspan=2, sticky=tk.W, pady=2)

        tk.Label(plot_controls_frame, text="Y-min (µV):", bg=DARK_BG, fg=DARK_FG).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.y_min_entry = tk.Entry(plot_controls_frame, width=10, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.y_min_entry.grid(row=4, column=1, sticky=tk.W, pady=2)

        tk.Label(plot_controls_frame, text="Y-max (µV):", bg=DARK_BG, fg=DARK_FG).grid(row=4, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.y_max_entry = tk.Entry(plot_controls_frame, width=10, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.y_max_entry.grid(row=4, column=3, columnspan=2, sticky=tk.W, pady=2)

        tk.Label(plot_controls_frame, text="V-Lines (ms, comma-sep):", bg=DARK_BG, fg=DARK_FG).grid(row=5, column=0, sticky=tk.W, pady=2)
        self.v_lines_entry = tk.Entry(plot_controls_frame, width=40, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.v_lines_entry.grid(row=5, column=1, columnspan=3, sticky=tk.EW, pady=2)

        tk.Label(plot_controls_frame, text="Plot Title:", bg=DARK_BG, fg=DARK_FG).grid(row=6, column=0, sticky=tk.W, pady=2)
        self.plot_title_entry = tk.Entry(plot_controls_frame, width=40, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.plot_title_entry.insert(0, "ERP Waveform")
        self.plot_title_entry.grid(row=6, column=1, columnspan=3, sticky=tk.EW, pady=2)

        tk.Button(plot_controls_frame, text="Update Plot", command=self._update_plot_display, bg=DARK_BUTTON_BG, fg=DARK_BUTTON_FG, activebackground=DARK_ACTIVE_BUTTON_BG, activeforeground=DARK_FG, relief=tk.FLAT, padx=5).grid(row=7, column=1, pady=10)
        tk.Button(plot_controls_frame, text="Save Plot", command=self._export_plot, bg=DARK_BUTTON_BG, fg=DARK_BUTTON_FG, activebackground=DARK_ACTIVE_BUTTON_BG, activeforeground=DARK_FG, relief=tk.FLAT, padx=5).grid(row=7, column=2, padx=(5,0), pady=10)
        
        tk.Label(plot_controls_frame, text="Export DPI:", bg=DARK_BG, fg=DARK_FG).grid(row=7, column="3", sticky=tk.E, padx=(10,0), pady=2)
        self.dpi_entry = tk.Entry(plot_controls_frame, width=5, bg=DARK_ENTRY_BG, fg=DARK_ENTRY_FG, insertbackground=DARK_INSERT_BG, relief=tk.FLAT)
        self.dpi_entry.insert(0, "300")
        self.dpi_entry.grid(row=7, column="4", sticky=tk.W, padx=(0,5), pady=2)

        plot_controls_frame.columnconfigure(1, weight=1)
        plot_controls_frame.columnconfigure(3, weight=1)

        # --- Plot Display Section (Default/Light Theme for Plot) ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6)) # Matplotlib default light theme
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_display_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        # No specific bg for canvas_widget, it will blend with plot_display_frame's default
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self._initialize_plot()


    def _initialize_plot(self):
        """Sets up the initial empty plot (light theme)."""
        self.ax.clear()
        # Labels and title will use Matplotlib's default (light theme) colors
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Amplitude (µV)')
        self.ax.set_title('Load data and configure plot options')
        # Grid will use Matplotlib's default light theme grid settings
        self.ax.grid(self.gridlines_var.get(), linestyle='--', linewidth=0.5)
        self.canvas.draw()

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
            self.root.destroy()

    def _browse_input_erp(self):
        filename = filedialog.askopenfilename(
            title="Select Input ERP file",
            filetypes=[("ERPLAB files", "*.erp"), ("All files", "*.*")]
        )
        if filename:
            self.input_erp_entry.delete(0, tk.END)
            self.input_erp_entry.insert(0, filename)
            default_output_fif = os.path.splitext(filename)[0] + ".fif"
            self.output_fif_entry.delete(0, tk.END)
            self.output_fif_entry.insert(0, default_output_fif)

    def _browse_output_fif(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output FIF as",
            defaultextension=".fif",
            filetypes=[("MNE FIF files", "*.fif"), ("All files", "*.*")]
        )
        if filename:
            self.output_fif_entry.delete(0, tk.END)
            self.output_fif_entry.insert(0, filename)
            
    def _browse_and_load_fif(self):
        filename = filedialog.askopenfilename(
            title="Select FIF file to load",
            filetypes=[("MNE FIF files", "*.fif"), ("All files", "*.*")]
        )
        if filename:
            self.current_fif_file.set(filename)
            self._load_fif_data(filename)

    def _load_fif_data(self, fif_file_path):
        if not fif_file_path or not os.path.exists(fif_file_path):
            messagebox.showerror("Error", f"FIF file not found: {fif_file_path}")
            self.evokeds_data = None
            self._initialize_plot()
            return
        try:
            self.evokeds_data = mne.read_evokeds(fif_file_path, verbose=False)
            messagebox.showinfo("Success", f"Loaded {len(self.evokeds_data)} evoked condition(s) from '{os.path.basename(fif_file_path)}'.")
            self._update_plot_display()
        except Exception as e:
            self.evokeds_data = None
            messagebox.showerror("FIF Load Error", f"Failed to load .fif file: {e}\nFile: {fif_file_path}")
            self._initialize_plot()

    def _run_conversion(self):
        input_file = self.input_erp_entry.get()
        output_file = self.output_fif_entry.get()

        if not input_file or not os.path.exists(input_file):
            messagebox.showerror("Error", "Input .erp file is invalid or not specified.")
            return
        if not output_file:
            messagebox.showerror("Error", "Output .fif file path not specified.")
            return

        success, message = erp_to_fif(input_file, output_file)
        if success:
            messagebox.showinfo("Conversion Success", message)
            self.current_fif_file.set(output_file)
            self._load_fif_data(output_file)
        else:
            messagebox.showerror("Conversion Failed", message)
            
    def _update_plot_display(self):
        if self.evokeds_data is None:
            if self.current_fif_file.get():
                messagebox.showwarning("No Data", "FIF data could not be loaded. Please check the file or load another.")
            else:
                messagebox.showwarning("No Data", "No FIF data loaded. Please load a .fif file first.")
            self._initialize_plot()
            return

        channels_str = self.channels_entry.get()
        bins_str = self.bins_entry.get()
        plot_title = self.plot_title_entry.get()
        do_avg_channels = self.avg_channels_var.get()
        do_avg_bins = self.avg_bins_var.get()
        show_grid = self.gridlines_var.get()
        v_lines_str = self.v_lines_entry.get()

        y_min_str = self.y_min_entry.get()
        y_max_str = self.y_max_entry.get()
        y_limits = None

        try:
            time_start_ms = float(self.time_start_entry.get())
            time_end_ms = float(self.time_end_entry.get())
            if y_min_str and y_max_str:
                y_min_val = float(y_min_str)
                y_max_val = float(y_max_str)
                if y_min_val < y_max_val:
                    y_limits = (y_min_val, y_max_val)
                else:
                    messagebox.showwarning("Input Warning", "Y-min must be less than Y-max. Y-axis will autoscale.")
            elif y_min_str or y_max_str:
                messagebox.showwarning("Input Warning", "Both Y-min and Y-max must be specified for custom Y-axis scale. Y-axis will autoscale.")
        except ValueError:
            messagebox.showerror("Input Error", "Time window and Y-axis scale values must be numeric.")
            return

        if not channels_str:
            messagebox.showerror("Input Error", "Channel(s) are required for plotting.")
            return
        if not bins_str:
            messagebox.showerror("Input Error", "Bin Indices are required for plotting.")
            return
            
        user_v_lines_ms = []
        if v_lines_str:
            try:
                user_v_lines_ms = [float(val.strip()) for val in v_lines_str.split(',') if val.strip()]
            except ValueError:
                messagebox.showwarning("Input Warning", "V-Lines contains non-numeric values. Invalid entries will be ignored.")
                valid_v_lines = []
                for val in v_lines_str.split(','):
                    if val.strip():
                        try:
                            valid_v_lines.append(float(val.strip()))
                        except ValueError:
                            pass 
                user_v_lines_ms = valid_v_lines

        try:
            target_channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]
            target_bin_indices = [int(b.strip()) - 1 for b in bins_str.split(',') if b.strip()]
            if not target_channels:
                messagebox.showerror("Input Error", "No valid channels specified.")
                return
            if not target_bin_indices:
                messagebox.showerror("Input Error", "No valid bin indices specified.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Invalid channel or bin format. Use comma-separated values (bins are 1-based).")
            return

        self.ax.clear() # Clears the plot for redrawing
        waveforms_by_main_group = {}
        found_any_data_at_all = False

        # Data processing logic (remains the same)
        if do_avg_channels:
            avg_channels_data_for_bins = []
            for bin_idx in target_bin_indices:
                if not (0 <= bin_idx < len(self.evokeds_data)):
                    print(f"Warning: Bin index {bin_idx + 1} is out of range. Skipping.")
                    continue
                ev_bin = self.evokeds_data[bin_idx]
                current_bin_channel_numpy_data = []
                current_bin_times = None
                for ch_name in target_channels:
                    if ch_name in ev_bin.ch_names:
                        ch_idx_ev = ev_bin.ch_names.index(ch_name)
                        numpy_data = ev_bin.data[ch_idx_ev] * 1e6
                        current_bin_channel_numpy_data.append(numpy_data)
                        if current_bin_times is None:
                            current_bin_times = ev_bin.times * 1000
                        found_any_data_at_all = True
                    else:
                        print(f"Warning: Channel '{ch_name}' not found in {ev_bin.comment}. Skipping for this bin's channel average.")
                if current_bin_channel_numpy_data:
                    mean_data_for_bin = np.mean(np.array(current_bin_channel_numpy_data), axis=0)
                    avg_channels_data_for_bins.append({'data': mean_data_for_bin, 'times': current_bin_times, 'bin_comment': ev_bin.comment})
            if avg_channels_data_for_bins:
                waveforms_by_main_group['Avg Channels'] = avg_channels_data_for_bins
        else:
            for ch_name in target_channels:
                channel_specific_bin_data = []
                for bin_idx in target_bin_indices:
                    if not (0 <= bin_idx < len(self.evokeds_data)):
                        print(f"Warning: Bin index {bin_idx + 1} is out of range. Skipping.")
                        continue
                    ev_bin = self.evokeds_data[bin_idx]
                    if ch_name in ev_bin.ch_names:
                        ch_idx_ev = ev_bin.ch_names.index(ch_name)
                        numpy_data = ev_bin.data[ch_idx_ev] * 1e6
                        times_data = ev_bin.times * 1000
                        channel_specific_bin_data.append({'data': numpy_data, 'times': times_data, 'bin_comment': ev_bin.comment})
                        found_any_data_at_all = True
                    else:
                        print(f"Warning: Channel '{ch_name}' not found in {ev_bin.comment}. Skipping.")
                if channel_specific_bin_data:
                    waveforms_by_main_group[ch_name] = channel_specific_bin_data

        plotted_something = False
        # Plotting waveforms (Matplotlib default light theme colors will be used for lines)
        for main_label_part, list_of_waveforms in waveforms_by_main_group.items():
            if not list_of_waveforms:
                continue
            if do_avg_bins:
                data_arrays_for_bin_avg = [item['data'] for item in list_of_waveforms]
                if data_arrays_for_bin_avg:
                    min_len = min(len(d) for d in data_arrays_for_bin_avg)
                    data_arrays_for_bin_avg = [d[:min_len] for d in data_arrays_for_bin_avg]
                    final_averaged_data = np.mean(np.array(data_arrays_for_bin_avg), axis=0)
                    times_for_plot = list_of_waveforms[0]['times'][:min_len]
                    plot_label = f'{main_label_part} - Avg Bins'
                    self.ax.plot(times_for_plot, final_averaged_data, label=plot_label)
                    plotted_something = True
            else:
                for waveform_item in list_of_waveforms:
                    times_for_plot = waveform_item['times']
                    plot_label = f'{main_label_part} - {waveform_item["bin_comment"]}'
                    self.ax.plot(times_for_plot, waveform_item['data'], label=plot_label)
                    plotted_something = True
        
        # Plot title and legend (Matplotlib default light theme colors)
        if not plotted_something and found_any_data_at_all:
            self.ax.set_title(f"{plot_title} (Check averaging options or selections)")
        elif not plotted_something and not found_any_data_at_all:
            self.ax.set_title(f"{plot_title} (No data for selection)")
        else:
            self.ax.set_title(plot_title)
            self.ax.legend(loc='best', fontsize='small')

        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Amplitude (µV)')
        self.ax.set_xlim([time_start_ms, time_end_ms])
        if y_limits:
            self.ax.set_ylim(y_limits)
        
        # Grid (Matplotlib default light theme color/style)
        self.ax.grid(show_grid, linestyle='--', linewidth=0.5)
        # Zero line (black for light theme)
        #self.ax.axvline(0, color=DEFAULT_PLOT_ZERO_LINE_COLOR, linestyle='--', linewidth=0.7)

        # Add user-defined v-lines (cyan)
        for v_line_ms in user_v_lines_ms:
            self.ax.axvline(v_line_ms, color=USER_VLINE_COLOR, linestyle=':', linewidth=1)
            
        self.canvas.draw()

    def _export_plot(self):
        if not self.ax.has_data():
            messagebox.showwarning("Export Error", "No plot to export. Please generate a plot first.")
            return
            
        try:
            dpi = int(self.dpi_entry.get())
            if dpi <= 0:
                raise ValueError("DPI must be positive.")
        except ValueError:
            messagebox.showerror("Input Error", "Invalid DPI value. Please enter a positive integer.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Plot As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("SVG files", "*.svg"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                # Save with the figure's current facecolor (default light for Matplotlib)
                current_fig_facecolor = self.fig.get_facecolor()
                self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=current_fig_facecolor)
                messagebox.showinfo("Export Success", f"Plot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save plot: {e}")

if __name__ == "__main__":
    main_window = tk.Tk()
    app = ErpProcessorApp(main_window)
    main_window.mainloop()