import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import tkinter as tk

# Define the STFT and inverse STFT functions
def stft(x, R, Nfft):
    hop_size = R//2
    w = np.hanning(R)
    stft_matrix = np.empty((Nfft//2 + 1, 1 + (len(x) - R) // hop_size), dtype=np.complex64)
    #print(stft_matrix.shape[1])
    for i in range(stft_matrix.shape[1]):
        segment = x[i * hop_size : i * hop_size + R]
        windowed_segment = segment * w
        stft_matrix[:, i] = np.fft.rfft(windowed_segment, Nfft)
    return stft_matrix

def inv_stft(X, R, N):
    hop_size = R//2
    w = np.hanning(R)
    x = np.zeros(N, dtype=np.float32)
    for i in range(X.shape[1]):
        segment = np.fft.irfft(X[:, i], R)
        x[i * hop_size : i * hop_size + R] += segment * w
    return x


# Define the parameters of the STFT
R = 512
Nfft = R

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                output=True,
                frames_per_buffer=R)

print("Start speaking")

# Create a Tkinter window
root = tk.Tk()
root.title("Robotization")
root.geometry("600x150")


tk.Label(root, text="").pack()

tk.Label(root, text="* Start speaking to hear the robotization effect *", font=("Helvetica", 15)).pack()

tk.Label(root, text="").pack()

# # Define a function to quit the program
# def quit_program():
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     root.destroy()

is_running = True

# Define a function to quit the program
def quit_program():
    global is_running
    is_running = False
    # Concatenate the input and output signals into arrays
    x_arr = np.concatenate(x_list)
    y2_arr = np.concatenate(y2_list)

    # Compute the STFT of the input and output signals
    X_arr = stft(x_arr, R, Nfft)
    Y2 = stft(y2_arr, R, Nfft)

    # Plot the input and output spectrograms
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    im1 = axs[0].imshow(np.abs(X_arr), origin='lower', aspect='auto')
    axs[0].set_title("Input Spectrogram")
    im2 = axs[1].imshow(np.abs(Y2), origin='lower', aspect='auto')
    axs[1].set_title("Output Spectrogram")

    # Add axis labels
    axs[0].set_ylabel("Frequency")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlabel("Time")

    # Create a color bar
    cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.4, pad=0.01)
    cbar.ax.set_ylabel('Magnitude')

    # Adjust the position of the color bar
    #fig.subplots_adjust(right=0.6)

    plt.show()


tk.Label(root, text="To try other effects & see spectrograms ... ").pack()
# Create a quit button
quit_button = tk.Button(root, text="Quit", command=quit_program, fg="red", bg="white", font=("Helvetica", 20))
quit_button.pack()


x_list = []
X_list = []
X2_list = []
y2_list = []

# Start processing audio in real-time
while is_running:
    root.update()   # Update the Tkinter window
    # Read a chunk of audio from the microphone
    input_data = stream.read(R, exception_on_overflow=False)
    # Convert the input data from bytes to a numpy array of floats
    x = np.frombuffer(input_data, dtype=np.float32)
    # Apply a threshold to the input signal
    if np.max(np.abs(x)) < 0.01:
        x = np.zeros_like(x)  # Set the input to zero if it's too faint
    # Compute the STFT of the input signal
    X = stft(x, R, Nfft)
    # Calculate the magnitude spectrum of the STFT by taking the absolute value of the complex STFT coefficients	
    X2 = np.abs(X)
    # Reconstruct the time-domain signal from the modified STFT coefficients
    y2 = inv_stft(X2, R, len(x))
    # Normalize the output signal
    if np.max(np.abs(y2)) > 0:
        y2 = y2 / np.max(np.abs(y2))
    y2 *= 0.08 # Apply the scaling factor from the slider

    y2_list.append(y2)
    x_list.append(x)


    # Convert the output signal to bytes
    output_data = y2.astype(np.float32).tobytes()
    # Write the output data to the output stream to play the processed audio back to the user
    stream.write(output_data)

# Stop the audio stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
# Close the Tkinter window
root.destroy()

