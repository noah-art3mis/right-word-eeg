import mne
import pandas as pd


def main():
    FILENAME = "EEG_Cat_Study5_II_S1.bdf"
    raw = mne.io.read_raw_bdf(FILENAME, preload=True)
    raw.plot()

    # Get EEG channel names
    channel_names = raw.ch_names
    print("Channels:", channel_names)

    # Get sampling frequency
    sfreq = raw.info["sfreq"]
    print("Sampling Frequency:", sfreq, "Hz")

    # Apply bandpass filter (1-40 Hz)
    raw.filter(l_freq=1.0, h_freq=40.0)

    save_to_csv(raw, "eeg_data.csv")


def save_to_csv(_raw, filename: str):

    # Convert EEG data to NumPy array
    eeg_data, times = _raw[:]

    # Create DataFrame
    df = pd.DataFrame(eeg_data.T, columns=_raw.ch_names)
    df["Time"] = times

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"EEG data saved as {filename}")


if __name__ == "__main__":
    main()
