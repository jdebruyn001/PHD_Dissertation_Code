import pandas as pd
import pyttsx3
import os

# Function to convert text to speech and save as MP3 using pyttsx3
def text_to_speech_pyttsx3(text, filename):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Load the Excel file
file_path = r'C:/Users/jaco/OneDrive/Desktop/PHD/Python/Data/MULTIWOZ Data.xlsx'
data = pd.read_excel(file_path)

# Directory to save the MP3 files
output_directory = r'C:/Users/jaco/OneDrive/Desktop/PHD/Python/MP3_Files'

# Process the dataframe
for index, row in data.iterrows():
    if row['Text vs Voice'] == 'Voice':
        filename = os.path.join(output_directory, str(row['Steps']) + '.mp3')
        text_to_speech_pyttsx3(row['Value.log.text'], filename)

print("Conversion completed. Files are saved in:", output_directory)
