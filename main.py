import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

END_HEADER_FLAG = "endheader"

# For this funtion we consider that the "Data" directory and "main.py" file are located in the same directory
def load_data(emg_file, angle_file):
    """
    Load the EMG and knee angle data from the provided files.

    Args:
    - emg_file (str): Path to the EMG .sto file.
    - angle_file (str): Path to the knee angle .mot file.

    Returns:
    - emg_data (list): List containing the loaded EMG data.
    - angle_data (list): List containing the loaded knee angle data.
    """

    # TODO: Implement file loading logic here (e.g., use pandas or numpy to read the files)

    def  file_reader(file_path):
        line_count = 0
        header = {}
        with open(file_path, 'r') as f:

            for line_str in f:
                line_count += 1
                if END_HEADER_FLAG in line_str:
                    line_count += 1
                    break
                else:
                    if "=" in line_str:
                        try:
                            key,value = line_str.strip().split("=")
                            header[key] = value                         # Might come in handy later :)
                        except ValueError:
                            print("Header formating error\n",ValueError) 
                
        data = np.loadtxt(file_path, skiprows=line_count)
        return data, header
    

    emg_data  , emg_header= file_reader(emg_file)
    angle_data, angle_header = file_reader(angle_file)

    return emg_data, angle_data

def plot_raw_data(emg_data, angle_data):
    """
    Plot the raw EMG and knee angle data.

    Args:
    - emg_data (list or pd.DataFrame): Data containing the raw EMG data.
    - angle_data (list or pd.DataFrame): Data containing the raw knee angle data.

    The function should display the plot with both the EMG and knee angle data.
    Ensure that you use 'plt.show()' to visualize the plot when the code is run.
    """

    # TODO: Implement plotting here
    
    mv_emg_data = emg_data[:, 1] * 1e3  # Convert from V to mV

    _, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(angle_data[:, 0], angle_data[:, 1], 'r-', label="Knee Angle (°)",linewidth=1)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Knee Angle (°)", color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(emg_data[:, 0], mv_emg_data, 'g-', label="EMG (mV)",linewidth=0.8)
    ax2.set_ylabel("EMG (mV)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax1.grid(True)
    # plt.ylim(-10, 1000) Useful when comparing different EMG signals
    plt.title("Knee Angle and EMG Signal")

    plt.show()


def preprocess_emg(emg_data):
    """
    Preprocess the raw EMG data (filter, rectify, and smooth).

    Args:
    - emg_data (list or pd.DataFrame): List or DataFrame containing the raw EMG data.

    Returns:
    - emg_preprocessed (list or pd.DataFrame): Preprocessed EMG data after filtering, rectifying, and smoothing.
    """

    # Initialise a variable for the preprocessed EMG data
    emg_preprocessed = []
    
    # TODO: Implement the preprocessing steps:

    # 1. Filter the raw EMG data to remove noise (e.g., bandpass filter).
    low_cutoff_freq  = 10
    high_cutoff_freq = 450
    order = 2
    sample_freq = int( 1/(emg_data[1][0]-emg_data[0][0]) )

    mv_emg_preprocessed = emg_data[:, 1] * 1e3

    b, a = butter(order, [low_cutoff_freq, high_cutoff_freq], btype='bandpass', fs=sample_freq)
    filtered_signal = filtfilt(b, a, mv_emg_preprocessed) 
    #show_figure(  emg_data[:,0],  filtered_signal,"EMG Band-pass Filter","Eletric potential","Volts", linewidth=0.2)
    
    # 2. Rectify the EMG signal (take the absolute value).
    rectified_signal = np.abs(filtered_signal)
    #show_figure(  emg_data[:,0], rectified_signal,"EMG Rectified","Eletric potential","Volts", linewidth=0.2)

    # 3. Smooth the EMG signal (e.g., moving average or low-pass filter).
    window_size = 100
    smoothed_signal = np.convolve(rectified_signal, np.ones(window_size)/window_size, mode='same')
    #show_figure(  emg_data[:,0],  smoothed_signal,"EMG Smoothed","Eletric potential","mV", linewidth=0.6)

    # Plot raw EMG and smoothed signal
    _, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(emg_data[:, 0], emg_data[:, 1], label="Raw EMG", alpha=0.8,linewidth=0.8, color = "g")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Raw EMG [mV]", color="g")
    ax1.tick_params(axis='y', labelcolor="g")

    ax2 = ax1.twinx()
    ax2.plot(emg_data[:, 0], smoothed_signal, label="Smoothed EMG", linewidth=0.6, color = "orange")
    ax2.set_ylabel("Smoothed EMG [mV]", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    ax1.grid(True)
    plt.title("Raw X Smoothed EMG Signal")

    plt.show()


    emg_preprocessed = np.column_stack((emg_data[:,0],smoothed_signal))

    return emg_preprocessed


def calculate_fatigue_metrics(emg_data, angle_data):
    """
    Calculate fatigue metrics from the EMG and knee angle data, split into segments.

    Args:
    - emg_data (pd.DataFrame or dict-like): Must have at least the key 'vasmed_l'.
    - angle_data (pd.DataFrame or dict-like): Must have at least the key 'knee_angle'.

    Returns:
    - emg_fatigue (np.ndarray): 2D array of shape, where each row contains [RMS, MAV] for that segment.
    - angle_fatigue (np.ndarray): 2D array of shape, where each row contains [ROM, mean_change] for that segment.
    """
    
    def rms(signal):
        return np.sqrt(np.mean(signal ** 2)) # Root Mean Squared

    def mav(signal):
        return np.mean(np.abs(signal)) # Mean Absolute Value
    
    def rom(angle_segment):
        return np.max(angle_segment) - np.min(angle_segment) # Range of Motion
    
    def mnc(signal):                
        return np.mean(np.abs(np.diff(signal)))  # Mean Change

    def detect_emg_segment(signal):
        segment_list = []
        contractions = 10
        peaks,_ = find_peaks(signal, distance=1500)

        if len(peaks) > contractions:
            sorted_peaks = np.argsort(signal[peaks])[-contractions:]
            peaks = np.sort(peaks[sorted_peaks])

        index_list = [(peaks[i+1]+peaks[i])//2 for i in range(len(peaks)-1)]
        index_list.insert(0,0)    
        index_list.append(len(signal)-1) 
        for i in range(len(index_list)-1):
            segment_list.append(signal[index_list[i] : index_list[i+1]])

        return segment_list
    
    def detect_ang_segment(signal):
        segment_list = []
        contractions = 10
        peaks,_ = find_peaks(signal, distance=120, prominence=0.5)

        if len(peaks) > contractions+1:
            sorted_peaks = np.argsort(signal[peaks])[-contractions:]
            peaks = np.sort(peaks[sorted_peaks])

        segments_index = [i for i in peaks] # Makes peaks into a python list for ease of operation
        segments_index.insert(0,0)  
        for i in range(len(segments_index)-1):
            segment_list.append(signal[segments_index[i] : segments_index[i+1]])


        return segment_list
    



    
    # Initialise variables for storing the fatigue metrics
    emg_fatigue = []
    angle_fatigue = []

    # TODO: Implement the calculation of fatigue metrics:
    
    # 1. Calculate RMS or MAV for the EMG data to estimate muscle fatigue.
    full_emg_rms = rms(emg_data["vasmed_l"])
    full_emg_mav = mav(emg_data["vasmed_l"])
    emg_segment_list = detect_emg_segment(emg_data["vasmed_l"])
    emg_segment_rms = [rms(seg) for seg in emg_segment_list]
    emg_segment_mav = [mav(seg) for seg in emg_segment_list]
    emg_fatigue = np.array( [[rms, mav] for rms, mav in zip(emg_segment_rms, emg_segment_mav)] )    

    # 2. Calculate changes in knee angle (e.g., range of motion) to estimate joint fatigue.
    max_ang  = np.max(angle_data["knee_angle"])
    ang_segment_list = detect_ang_segment(angle_data["knee_angle"])
    ang_segment_rom = [rom(seg) for seg in ang_segment_list]
    ang_segment_mnc = [mnc(seg) for seg in ang_segment_list]
    angle_fatigue = np.array( [[rom, mnc] for rom, mnc in zip(ang_segment_rom, ang_segment_mnc)] )    


    return emg_fatigue, angle_fatigue

def plot_fatigue_metrics(emg_fatigue, angle_fatigue):
    """
    Plot the EMG and angle fatigue metrics across segments.

    Args:
        emg_fatigue (np.ndarray): 2D array
                                  emg_fatigue[:, 0] -> EMG RMS per segment
                                  emg_fatigue[:, 1] -> EMG MAV per segment
        angle_fatigue (np.ndarray): 2D array
                                    angle_fatigue[:, 0] -> Angle ROM per segment
                                    angle_fatigue[:, 1] -> Mean change per segment
    """

    # TODO: Implement plotting here
    num_segments = len(emg_fatigue)
    segments = np.arange(1, num_segments+1)  

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(segments, emg_fatigue[:, 0], label="EMG RMS", marker="o", color="blue")
    ax1.plot(segments, emg_fatigue[:, 1], label="EMG MAV", marker="s", color="orange")
    ax1.set_title("EMG Fatigue Metrics Across Segments")
    ax1.set_xlabel("Segment")
    ax1.set_ylabel("Amplitude")
    ax1.set_xticks(segments)
    ax1.legend()
    ax1.grid()

    ax2_rom = ax2 
    ax2_mean_change = ax2.twinx() 

    ax2_rom.plot(segments, angle_fatigue[:, 0], label="ROM", marker="o", color="green")
    ax2_rom.set_ylabel("ROM (degrees)", color="green")
    ax2_rom.tick_params(axis="y", labelcolor="green")

    ax2_mean_change.plot(segments, angle_fatigue[:, 1], label="Mean Change", marker="s", color="red")
    ax2_mean_change.set_ylabel("Mean Change (degrees)", color="red")
    ax2_mean_change.tick_params(axis="y", labelcolor="red")

    ax2.set_title("Angle Fatigue Metrics Across Segments")
    ax2.set_xlabel("Segment")
    ax2.set_xticks(segments)
    ax2.grid()

    plt.tight_layout()

    # Display the plots
    plt.show()




################## additional functions ##################

def show_figure(x_axis,y_axis,title, label, unit, linewidth = 1, c = "g"):
    plt.figure(figsize=(8, 6))

    plt.plot(x_axis,y_axis, linestyle='-', color=c, label = label,linewidth=linewidth)

    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel(f"{label} [{unit}]")
    plt.grid()
    plt.legend()

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define the paths to your EMG and knee angle files
    emg_file   =   "Data/ID01_emg_0.sto"
    angle_file = "Data/ID01_angle_0.mot"

    # Step 1: Load the data
    emg_data, angle_data = load_data(emg_file, angle_file)

    ############################### separate data ploting ###############################
    show_figure(  emg_data[:,0],  emg_data[:,1],"Vastus Medialis EMG signal","Eletric potential","Volts",linewidth=0.8)
    show_figure(angle_data[:,0],angle_data[:,1],"Knee Joint Angle","Angle","°",linewidth=1, c='r')
    #####################################################################################

    # Step 2: Plot the raw data
    plot_raw_data(emg_data, angle_data)

    # Step 3: Preprocess the EMG data (filter, rectify, and smooth)
    emg_preprocessed = preprocess_emg(emg_data)

    emg_dic = {"time": emg_preprocessed[:,0],"vasmed_l":emg_preprocessed[:,1]}
    ang_dic = {"time":       angle_data[:,0],"knee_angle":    angle_data[:,1]}

    # Step 4: Calculate fatigue metrics
    emg_fatigue, angle_fatigue = calculate_fatigue_metrics(emg_dic,ang_dic)

    # Step 5: Plot the fatigue metrics
    plot_fatigue_metrics(emg_fatigue, angle_fatigue)


