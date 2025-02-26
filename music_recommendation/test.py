import tkinter as tk
from tkinter import filedialog, messagebox, Listbox
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess


class MeasureZoomVisualizer:
    """
    Interactive visualization for waveform with measure zoom functionality.
    """
    def __init__(self, y, sr, beat_times, measures):
        self.y = y  # Audio signal
        self.sr = sr  # Sampling rate
        self.beat_times = beat_times  # Beat timestamps
        self.measures = measures  # Measure timestamps

        # Initialize the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(14, 4))
        self.plot_waveform()

        # Connect click events for interactive zoom
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def plot_waveform(self, measure_idx=None):
        """Plot the waveform and annotate measures."""
        self.ax.clear()
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.6, ax=self.ax)

        # Highlight specific measure if zoomed in
        if measure_idx is not None:
            measure_start = self.measures[measure_idx][0]
            measure_end = self.measures[measure_idx][-1]
            self.ax.set_xlim(measure_start, measure_end)
            self.ax.set_title(f"Zoomed into Measure {measure_idx + 1}")
        else:
            self.ax.set_title("Waveform with Detected Beats and Measures")

        # Plot beats and annotate measures
        self.ax.vlines(self.beat_times, -1, 1, color='r', alpha=0.75, label='Beats')
        for i, measure in enumerate(self.measures):
            self.ax.text(measure[0], 1, f'M{i + 1}', color='blue', fontsize=8, verticalalignment='bottom')

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()

    def on_click(self, event):
        """Handle click events for measure zooming."""
        if event.inaxes != self.ax:
            return

        # Check if the click falls within a measure range
        for i, measure in enumerate(self.measures):
            measure_start = measure[0]
            measure_end = measure[-1]
            if measure_start <= event.xdata <= measure_end:
                print(f"Clicked on Measure {i + 1}")
                self.plot_waveform(measure_idx=i)
                return


class SongSeparatorApp:
    """
    Main application for separating tracks and analyzing a specific track.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Track Separator and Analyzer")
        self.root.geometry("600x500")

        # Variables
        self.song_path = None
        self.output_folder = None
        self.separated_files = []

        # Set up the GUI layout
        self.setup_gui()

    def setup_gui(self):
        """Set up the GUI layout."""
        tk.Label(self.root, text="Track Separator and Analyzer", font=("Arial", 16)).pack(pady=10)

        # Input song selection
        tk.Label(self.root, text="Choose a Song File:", font=("Arial", 12)).pack(pady=5)
        self.song_path_entry = tk.Entry(self.root, width=50, state='readonly')
        self.song_path_entry.pack(pady=5)
        tk.Button(self.root, text="Browse", command=self.select_song).pack(pady=5)

        # Output folder selection
        tk.Label(self.root, text="Output Folder for Separated Tracks:", font=("Arial", 12)).pack(pady=5)
        self.output_folder_entry = tk.Entry(self.root, width=50, state='readonly')
        self.output_folder_entry.pack(pady=5)
        tk.Button(self.root, text="Choose Folder", command=self.select_output_folder).pack(pady=5)

        # Buttons
        tk.Button(self.root, text="Separate Tracks", command=self.separate_tracks, bg="green", fg="white").pack(pady=10)

        # Separated tracks display
        tk.Label(self.root, text="Separated Tracks:", font=("Arial", 12)).pack(pady=5)
        self.track_listbox = Listbox(self.root, width=60, height=10)
        self.track_listbox.pack(pady=5)
        tk.Button(self.root, text="Analyze Selected Track", command=self.analyze_selected_track, bg="blue", fg="white").pack(pady=10)

        # Status message
        self.status_label = tk.Label(self.root, text="", font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)

    def select_song(self):
        """Select a song file for separation."""
        self.song_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
        if self.song_path:
            self.song_path_entry.config(state='normal')
            self.song_path_entry.delete(0, tk.END)
            self.song_path_entry.insert(0, self.song_path)
            self.song_path_entry.config(state='readonly')

    def select_output_folder(self):
        """Select the output folder for separated tracks."""
        self.output_folder = filedialog.askdirectory()
        if self.output_folder:
            self.output_folder_entry.config(state='normal')
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, self.output_folder)
            self.output_folder_entry.config(state='readonly')

    def separate_tracks(self):
        """Run Spleeter to separate the tracks of the selected song."""
        if not self.song_path or not self.output_folder:
            messagebox.showerror("Error", "Please select both a song and an output folder.")
            return

        self.status_label.config(text="Processing... Please wait.", fg="blue")
        self.root.update()

        # Run Spleeter separation
        try:
            spleeter_command = [
                "spleeter", "separate",
                "-p", "spleeter:5stems",  # Separate into 5 stems (vocals, bass, drums, piano, other)
                "-o", self.output_folder,
                self.song_path
            ]
            subprocess.run(spleeter_command, check=True)
            self.status_label.config(text="Tracks separated successfully!", fg="green")
            messagebox.showinfo("Success", "Tracks were successfully separated.")
            self.display_separated_files()
        except Exception as e:
            self.status_label.config(text="Error during separation.", fg="red")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def display_separated_files(self):
        """Display the separated tracks in the listbox."""
        self.track_listbox.delete(0, tk.END)
        self.separated_files = []

        # Locate the folder containing the separated files
        output_dir = os.path.join(self.output_folder, os.path.splitext(os.path.basename(self.song_path))[0])
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith(".wav"):
                    full_path = os.path.join(output_dir, file)
                    self.separated_files.append(full_path)
                    self.track_listbox.insert(tk.END, file)

    def analyze_selected_track(self):
        """Analyze the selected track from the listbox."""
        selected_index = self.track_listbox.curselection()
        if not selected_index:
            messagebox.showerror("Error", "Please select a track to analyze.")
            return

        track_path = self.separated_files[selected_index[0]]
        try:
            result = self.analyze_track(track_path)
            MeasureZoomVisualizer(result["audio"], result["sample_rate"], result["beat_times"], result["measures"])
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")

    @staticmethod
    def analyze_track(file_path):
        """
        Analyze the track to extract tempo, beats, and measures.
        """
        print("Loading audio file...")
        y, sr = librosa.load(file_path, sr=None)

        print("Estimating tempo and beats...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Estimated Tempo: {tempo:.2f} BPM")

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        print("Segmenting into measures...")
        time_signature = 4  # Assume 4/4 time signature
        measures = [
            beat_times[i:i + time_signature]
            for i in range(0, len(beat_times) - time_signature, time_signature)
        ]
        print(f"Detected {len(measures)} measures.")

        return {
            "tempo": tempo,
            "beat_times": beat_times,
            "measures": measures,
            "audio": y,
            "sample_rate": sr,
        }


if __name__ == "__main__":
    root = tk.Tk()
    app = SongSeparatorApp(root)
    root.mainloop()
