import numpy as np
from scipy.fft import rfft, fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt

class FMCWVitalSignDetector:
    def __init__(self):
        # Radar system parameters
        self.ADC_SAMPLE_RATE = 1e6  # 1 MHz (but under-sampled)
        self.CHIRP_DURATION = 350e-6  # 350 μs
        self.SAMPLES_PER_CHIRP = 128  # 128 samples
        self.CHIRPS_PER_FRAME = 32  # 32 chirps
        self.BANDWIDTH = 4e9  # 4 GHz
        self.CARRIER_FREQ = 60e9  # 60 GHz
        self.FRAME_RATE = 10  # 10 FPS
        self.SPEED_OF_LIGHT = 3e8  # m/s

        # L-shaped array parameters
        self.NUM_RECEIVERS = 3  # 3 receivers
        self.RECEIVER_SPACING = 2.5e-3  # 2.5 mm (λ/2 at 60 GHz)
        self.WAVELENGTH = self.SPEED_OF_LIGHT / self.CARRIER_FREQ  # ~5 mm
        self.RECEIVER_POSITIONS = np.array([
            [self.RECEIVER_SPACING, 0],  # Receiver 1: x-axis
            [0, 0],  # Receiver 2: origin
            [0, self.RECEIVER_SPACING]  # Receiver 3: y-axis
        ])  # L-shaped array in xy-plane

        # Derived parameters
        self.ACTUAL_SAMPLE_RATE = self.SAMPLES_PER_CHIRP / self.CHIRP_DURATION  # ~365.7 kHz
        self.RANGE_RESOLUTION = self.SPEED_OF_LIGHT / (2 * self.BANDWIDTH)  # 3.75 cm
        self.MAX_RANGE = (self.SPEED_OF_LIGHT * self.CHIRP_DURATION) / 2  # 52.5m theoretical
        self.MAX_UNAMBIGUOUS_RANGE = (self.SPEED_OF_LIGHT * self.SAMPLES_PER_CHIRP) / (4 * self.BANDWIDTH)  # 2.4m
        self.FRAME_DURATION = self.CHIRPS_PER_FRAME * self.CHIRP_DURATION  # 11.2 ms
        self.DOPPLER_RESOLUTION = self.SPEED_OF_LIGHT / (2 * self.CARRIER_FREQ * self.FRAME_DURATION)  # 0.223 m/s
        self.MAX_VELOCITY = self.SPEED_OF_LIGHT / (4 * self.CARRIER_FREQ * self.CHIRP_DURATION)  # 3.57 m/s

        # Vital sign parameters
        self.BREATHING_FREQ_RANGE = (0.1, 0.5)  # 6-30 breaths per minute
        self.HEARTBEAT_FREQ_RANGE = (0.8, 2.5)  # 48-150 beats per minute

        # Signal processing parameters
        self.window_size = 100  # frames for moving average
        self.phase_history = []
        self.range_bin_history = []
        self.doppler_bins = np.fft.fftshift(np.fft.fftfreq(self.CHIRPS_PER_FRAME, d=self.CHIRP_DURATION)) * \
                            (self.SPEED_OF_LIGHT / (2 * self.CARRIER_FREQ))  # speed m/s
        self.range_bins = np.arange(int(self.SAMPLES_PER_CHIRP / 2) + 1) * self.RANGE_RESOLUTION

        print("FMCW Radar Vital Sign Detection System Initialized (L-shaped Array)")
        print(f"Range Resolution: {self.RANGE_RESOLUTION * 100:.2f} cm")
        print(f"Max Unambiguous Range: {self.MAX_UNAMBIGUOUS_RANGE:.2f} m")
        print(f"Doppler Resolution: {self.DOPPLER_RESOLUTION:.3f} m/s")
        print(f"Frame Rate: {self.FRAME_RATE} Hz")
        print(f"Number of Receivers: {self.NUM_RECEIVERS}")
        print(f"Receiver Positions: {self.RECEIVER_POSITIONS}")

    def range_fft(self, chirp_data):
        """Perform range FFT on chirp data"""
        window = np.hanning(self.SAMPLES_PER_CHIRP)
        windowed_data = chirp_data * window
        range_fft = rfft(windowed_data.real, n=self.SAMPLES_PER_CHIRP)
        return range_fft

    def doppler_fft(self, range_data):
        """Perform Doppler FFT across chirps"""
        window = np.hanning(self.CHIRPS_PER_FRAME)
        windowed_data = range_data * window[:, np.newaxis]
        doppler_fft = fft(windowed_data, axis=0)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
        return doppler_fft

    def beamform_signals(self, receiver_data, azimuth=0, elevation=0):
        """Apply delay-and-sum beamforming for L-shaped array"""
        # receiver_data shape: [chirps, receivers, range_bins]
        num_chirps, num_receivers, num_range_bins = receiver_data.shape
        beamformed_data = np.zeros((num_chirps, num_range_bins), dtype=complex)

        # Convert angles to radians
        theta = np.deg2rad(azimuth)
        phi = np.deg2rad(elevation)

        # Direction vector (unit vector in 3D space)
        k = 2 * np.pi / self.WAVELENGTH
        direction_vector = np.array([
            np.sin(theta) * np.cos(phi),  # x-component
            np.sin(theta) * np.sin(phi),  # y-component
            np.cos(theta)  # z-component
        ])

        # Compute phase shifts for each receiver
        phase_shifts = np.zeros(num_receivers, dtype=complex)
        for rx_idx in range(num_receivers):
            # Dot product of receiver position and direction vector
            phase = k * np.dot(self.RECEIVER_POSITIONS[rx_idx], direction_vector[:2])  # 2D positions
            phase_shifts[rx_idx] = np.exp(-1j * phase)

        # Apply beamforming
        for chirp_idx in range(num_chirps):
            for range_idx in range(num_range_bins):
                beamformed_data[chirp_idx, range_idx] = np.sum(
                    receiver_data[chirp_idx, :, range_idx] * phase_shifts
                )

        return beamformed_data

    def extract_vital_signs(self, radar_data):
        """Extract vital signs from multi-receiver radar data"""
        num_frames = radar_data.shape[0]
        phase_differences = []
        target_range_bin = 25  # Fixed at ~0.9375 m

        for frame_idx in range(num_frames):
            frame_data = radar_data[frame_idx]  # Shape: [receivers, chirps, samples]

            # Perform range FFT for each receiver and chirp
            range_profile = np.zeros(
                (self.CHIRPS_PER_FRAME, self.NUM_RECEIVERS, int(self.SAMPLES_PER_CHIRP/2)+1), dtype=complex
            )
            for rx_idx in range(self.NUM_RECEIVERS):
                for chirp_idx in range(self.CHIRPS_PER_FRAME):
                    range_profile[chirp_idx, rx_idx] = self.range_fft(frame_data[rx_idx, chirp_idx])

            # Apply 2D beamforming
            beamformed_profile = self.beamform_signals(range_profile, azimuth=0, elevation=0)

            target_range_bin = np.argmax(np.mean(np.abs(beamformed_profile)[:, 10:], axis=0))+10
            print(range_profile.shape, beamformed_profile.shape, "shape", target_range_bin)
            # Extract phase from target range bin
            target_signal = beamformed_profile[:, target_range_bin]
            avg_phase = np.angle(np.mean(target_signal))
            phase_differences.append(avg_phase)

        return np.array(phase_differences)

    def filter_vital_signs(self, phase_data, fs):
        """Filter phase data to extract breathing and heartbeat"""
        sos_breathing = signal.butter(4, self.BREATHING_FREQ_RANGE, btype='bandpass', fs=fs, output='sos')
        breathing_signal = signal.sosfilt(sos_breathing, phase_data)
        sos_heartbeat = signal.butter(4, self.HEARTBEAT_FREQ_RANGE, btype='bandpass', fs=fs, output='sos')
        heartbeat_signal = signal.sosfilt(sos_heartbeat, phase_data)
        return breathing_signal, heartbeat_signal

    def estimate_rates(self, breathing_signal, heartbeat_signal, fs, window_duration=10):
        """Estimate breathing and heart rates"""
        window_samples = int(window_duration * fs)
        if len(breathing_signal) < window_samples:
            return None, None

        breathing_window = breathing_signal[-window_samples:]
        heartbeat_window = heartbeat_signal[-window_samples:]
        freq_bins = fftfreq(window_samples, 1 / fs)

        breathing_fft = np.abs(fft(breathing_window))
        breathing_freq_idx = np.argmax(breathing_fft[1:window_samples // 2]) + 1
        breathing_rate = freq_bins[breathing_freq_idx] * 60

        heartbeat_fft = np.abs(fft(heartbeat_window))
        heartbeat_freq_idx = np.argmax(heartbeat_fft[1:window_samples // 2]) + 1
        heart_rate = freq_bins[heartbeat_freq_idx] * 60

        return breathing_rate, heart_rate

    def process_frame(self, frame_data):
        """Process a single frame of multi-receiver radar data"""
        range_doppler_map = np.zeros(
            (self.CHIRPS_PER_FRAME, int(self.SAMPLES_PER_CHIRP/2)+1), dtype=complex
        )

        # Perform range FFT and beamforming
        range_profile = np.zeros(
            (self.CHIRPS_PER_FRAME, self.NUM_RECEIVERS, int(self.SAMPLES_PER_CHIRP/2)+1), dtype=complex
        )
        for rx_idx in range(self.NUM_RECEIVERS):
            for chirp_idx in range(self.CHIRPS_PER_FRAME):
                range_profile[chirp_idx, rx_idx] = self.range_fft(frame_data[rx_idx, chirp_idx])

        beamformed_profile = self.beamform_signals(range_profile, azimuth=0, elevation=0)
        range_doppler_map = beamformed_profile

        doppler_fft = self.doppler_fft(range_doppler_map)
        return np.abs(doppler_fft)

    def run_detection(self, radar_data):
        """Main detection pipeline"""
        print("Running vital sign detection with L-shaped array processing...")
        phase_data = self.extract_vital_signs(radar_data)
        fs = self.FRAME_RATE
        breathing_signal, heartbeat_signal = self.filter_vital_signs(phase_data, fs)
        breathing_rate, heart_rate = self.estimate_rates(breathing_signal, heartbeat_signal, fs)

        return {
            'phase_data': phase_data,
            'breathing_signal': breathing_signal,
            'heartbeat_signal': heartbeat_signal,
            'breathing_rate': breathing_rate,
            'heart_rate': heart_rate
        }

    def visualize_results(self, results, radar_data):
        """Visualize detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        t = np.arange(len(results['phase_data'])) / self.FRAME_RATE

        first_frame_rd = self.process_frame(radar_data[0])
        first_frame_rd[:, :10] = 0
        r = np.arange(len(self.range_bins)) * self.RANGE_RESOLUTION
        axes[0, 0].plot(r, np.mean(np.abs(first_frame_rd), axis=0))
        axes[0, 0].set_title('Beamformed Range-Doppler Map (First Frame, L-shaped Array)')
        axes[0, 0].set_xlabel('Range (m)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)

        axes[0, 1].plot(t, results['phase_data'])
        axes[0, 1].set_title('Beamformed Phase Data')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Phase (radians)')
        axes[0, 1].grid(True)

        axes[1, 0].plot(t, results['breathing_signal'])
        axes[1, 0].set_title(f'Breathing Signal (Rate: {results["breathing_rate"]:.1f} breaths/min)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True)

        axes[1, 1].plot(t, results['heartbeat_signal'])
        axes[1, 1].set_title(f'Heartbeat Signal (Rate: {results["heart_rate"]:.1f} beats/min)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

def main_vitalsign(file, start_time, end_time=None, sample_rate=10):
    dataset = np.load(file)
    data = dataset['arr_0']  # Expected shape: [frames, receivers, chirps, samples]
    dataset.close()
    detector = FMCWVitalSignDetector()
    start_ = int(start_time * sample_rate)
    end_ = int(end_time * sample_rate) if end_time is not None else -1
    radar_data = data[start_:end_]
    print(radar_data.shape, "input data shape")
    results = detector.run_detection(radar_data)
    print(f"\nDetection Results:")
    print(f"Breathing Rate: {results['breathing_rate']:.1f} breaths/min")
    print(f"Heart Rate: {results['heart_rate']:.1f} beats/min")
    print(f"Expected Breathing Rate: 18.0 breaths/min")
    print(f"Expected Heart Rate: 72.0 beats/min")
    detector.visualize_results(results, radar_data)

if __name__ == "__main__":
    import os
    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "dataset", "vitalsign", "0619")
    save_file = os.path.join(folder_path, "vitalsign_0619_1.npz")
    start_time = 20
    end_time = 80
    main_vitalsign(save_file, start_time, end_time)
    print(save_file, "file")