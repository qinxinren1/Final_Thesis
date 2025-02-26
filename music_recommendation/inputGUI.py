import tkinter as tk
from tkinter import filedialog, messagebox, Listbox
import os
import subprocess
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import wave


class SongSeparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Song Separator with Spleeter")
        self.root.geometry("800x600")

        # Initialize pygame mixer for audio playback
        pygame.mixer.init()

        # Label for app title
        tk.Label(root, text="Song Separator", font=("Arial", 16)).pack(pady=10)

        # Song library selection
        tk.Label(root, text="Choose a Song from Library:", font=("Arial", 12)).pack(pady=5)
        self.song_path_entry = tk.Entry(root, width=50, state='readonly')
        self.song_path_entry.pack(pady=5)
        tk.Button(root, text="Browse", command=self.select_song).pack(pady=5)

        # Output folder selection
        tk.Label(root, text="Output Folder for Separated Tracks:", font=("Arial", 12)).pack(pady=5)
        self.output_folder_entry = tk.Entry(root, width=50, state='readonly')
        self.output_folder_entry.pack(pady=5)
        tk.Button(root, text="Choose Folder", command=self.select_output_folder).pack(pady=5)

        # Process button
        tk.Button(root, text="Separate Tracks", command=self.separate_tracks, bg="green", fg="white").pack(pady=20)

        # Status message
        self.status_label = tk.Label(root, text="", font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)

        # Separated tracks display
        tk.Label(root, text="Separated Tracks:", font=("Arial", 12)).pack(pady=10)

        # Listbox for displaying tracks
        self.track_listbox = Listbox(root, width=60, height=10)
        self.track_listbox.pack(pady=5)
        tk.Button(root, text="Show Waveform", command=self.show_waveform).pack(pady=5)
        tk.Button(root, text="Play Selected Track", command=self.play_track, bg="blue", fg="white").pack(pady=5)

        # Waveform display area
        self.figure = plt.Figure(figsize=(6, 2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # Initialize variables
        self.song_path = None
        self.output_folder = None
        self.separated_files = []

    def select_song(self):
        """Select a song file from the library."""
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
                "-p", "spleeter:2stems",  # 2 stems: vocals + accompaniment
                "-o", self.output_folder,
                self.song_path
            ]
            subprocess.run(spleeter_command, check=True)

            # Update status and display separated tracks
            self.status_label.config(text="Tracks separated successfully!", fg="green")
            messagebox.showinfo("Success", "The tracks were separated successfully.")
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
                    self.separated_files.append(os.path.join(output_dir, file))
                    self.track_listbox.insert(tk.END, file)

    def play_track(self):
        """Play the selected track from the listbox."""
        selected_index = self.track_listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "Please select a track to play.")
            return

        selected_file = self.separated_files[selected_index[0]]
        try:
            pygame.mixer.music.load(selected_file)
            pygame.mixer.music.play()
            self.status_label.config(text=f"Playing: {os.path.basename(selected_file)}", fg="blue")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while playing the track: {e}")

    def show_waveform(self):
        """Display the waveform of the selected track."""
        selected_index = self.track_listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "Please select a track to view its waveform.")
            return

        selected_file = self.separated_files[selected_index[0]]
        try:
            with wave.open(selected_file, 'r') as wav_file:
                # Read audio properties
                n_frames = wav_file.getnframes()
                n_channels = wav_file.getnchannels()
                framerate = wav_file.getframerate()
                duration = n_frames / framerate

                # Read frames and convert to numpy array
                frames = wav_file.readframes(n_frames)
                audio_data = np.frombuffer(frames, dtype=np.int16)

                # Handle stereo audio by averaging channels (down-mix to mono)
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                # Create time array
                time = np.linspace(0, duration, num=len(audio_data))

                # Plot waveform
                self.ax.clear()
                self.ax.plot(time, audio_data, color='blue')
                self.ax.set_title(f"Waveform of {os.path.basename(selected_file)}")
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel("Amplitude")
                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while displaying the waveform: {e}")


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SongSeparatorApp(root)
    root.mainloop()
