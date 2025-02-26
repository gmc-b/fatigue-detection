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

    emg_data = []
    angle_data = []

    # TODO: Implement file loading logic here (e.g., use pandas or numpy to read the files)

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

    # Display the plots
    # plt.show()  # This is necessary to visualize the plot when running the code


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

    # 2. Rectify the EMG signal (take the absolute value).

    # 3. Smooth the EMG signal (e.g., moving average or low-pass filter).

    return emg_preprocessed


def calculate_fatigue_metrics(emg_data, angle_data):
    """
    Calculate fatigue metrics from the EMG and knee angle data, split into segments.

    Args:
    - emg_data (pd.DataFrame or dict-like): Must have at least the key 'vasmed_l'.
    - angle_data (pd.DataFrame or dict-like): Must have at least the key 'knee_angle'.

    Returns:
    - emg_fatigue (np.ndarray): 2D array of shape, where each row contains [RMS, MAV] for that segment.
    - angle_fatigue (np.ndarray): 2D array of shap), where each row contains [ROM, mean_change] for that segment.
    """

    # Initialise variables for storing the fatigue metrics
    emg_fatigue = []
    angle_fatigue = []

    # TODO: Implement the calculation of fatigue metrics:

    # 1. Calculate RMS or MAV for the EMG data to estimate muscle fatigue.

    # 2. Calculate changes in knee angle (e.g., range of motion) to estimate joint fatigue.


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

    # Display the plots
    # plt.show()  # Use this to ensure the plot is shown when running the code


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define the paths to your EMG and knee angle files
    emg_file = "Data/ID01_emg_0.sto"
    angle_file = "Data/ID01_angle_0.mot"

    # Step 1: Load the data
    emg_data, angle_data = load_data(emg_file, angle_file)

    # Step 2: Plot the raw data
    plot_raw_data(emg_data, angle_data)

    # Step 3: Preprocess the EMG data (filter, rectify, and smooth)
    emg_preprocessed = preprocess_emg(emg_data)

    # Step 4: Calculate fatigue metrics
    emg_fatigue, angle_fatigue = calculate_fatigue_metrics(emg_data, angle_data)

    # Step 5: Plot the fatigue metrics
    plot_fatigue_metrics(emg_fatigue, angle_fatigue)


