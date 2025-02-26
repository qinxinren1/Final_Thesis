import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class MeasureZoomVisualizer:
    def __init__(self, y, sr, beat_times, measures):
        self.y = y  # Audio signal
        self.sr = sr  # Sampling rate
        self.beat_times = beat_times  # Beat timestamps
        self.measures = measures  # Measure timestamps
        self.current_zoom = None  # Current zoom state

        # Initialize the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(30, 3))
        self.plot_waveform()

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def plot_waveform(self, measure_idx=None):
        """Plot the waveform and annotate measures."""
        self.ax.clear()

        # Plot full waveform
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
        """Handle click events on the plot."""
        if event.inaxes != self.ax:
            return

        # Check if a measure was clicked
        for i, measure in enumerate(self.measures):
            measure_start = measure[0]
            measure_end = measure[-1]
            if measure_start <= event.xdata <= measure_end:
                print(f"Clicked on Measure {i + 1}")
                self.plot_waveform(measure_idx=i)
                return


def analyze_track(file_path):
    """
    Analyze a track to detect tempo, beats, time signature, and measures.
    Visualizes the waveform with beat and measure annotations.
    """
    # Step 1: Load the audio file
    print("Loading audio file...")
    y, sr = librosa.load(file_path, sr=None)  # Use original sampling rate

    # Step 2: Estimate Tempo and Beat Positions
    print("Estimating tempo and beats...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Estimated Tempo: {tempo:.2f} BPM")

    # Convert beat frames to time (seconds)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Step 3: Analyze Beat Intervals
    print("Analyzing beat intervals...")
    intervals = np.diff(beat_times)  # Differences between consecutive beats
    mean_interval = np.mean(intervals)
    print(f"Mean Beat Interval: {mean_interval:.2f} seconds")

    # Step 4: Estimate Time Signature (Simplistic)
    print("Estimating time signature...")
    time_signature = 4  # Assume 4/4 time signature for simplicity
    print(f"Estimated Time Signature: {time_signature}/4")

    # Step 5: Divide Song into Measures
    print("Segmenting into measures...")
    measures = []
    for i in range(0, len(beat_times) - time_signature, time_signature):
        measures.append(beat_times[i:i + time_signature])
    print(f"Total Measures Detected: {len(measures)}")

    # Step 6: Visualize with Interactive Zoom
    print("Visualizing waveform with interactive measure zoom...")
    MeasureZoomVisualizer(y, sr, beat_times, measures)

    # Return results
    return {
        "tempo": tempo,
        "time_signature": time_signature,
        "beat_times": beat_times,
        "measures": measures
    }


if __name__ == "__main__":
    # Specify the path to your .wav file
    file_path = "/Users/oltremare/Desktop/TUD/Final Thesis/spleeter/output/example1/drums.wav"

    # Analyze the track
    result = analyze_track(file_path)

    # Print the result summary
    print("\n=== Analysis Summary ===")
    print(f"Tempo: {result['tempo']:.2f} BPM")
    print(f"Time Signature: {result['time_signature']}/4")
    print(f"Total Beats: {len(result['beat_times'])}")
    print(f"Total Measures: {len(result['measures'])}")
    print("=========================")
