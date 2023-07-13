import numpy as np
import librosa
import time

class FeatureExtractor:
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.features_list = {
            "ae_mean": [], "ae_var": [], 
            "rms_mean": [], "rms_var": [], 
            "zcr_mean": [], "zcr_var": [], 
            "chroma_stft_mean": [], "chroma_stft_var": [], 
            "spec_centroid_mean": [], "spec_centroid_var": [], 
            "spec_cont_mean": [], "spec_cont_var": [],
            "spec_bw_mean": [], "spec_bw_var": [],
            "percep_mean": [], "percep_var": [], 
            "tempo_mean": [], "tempo_var": [], 
            "roll_off_mean": [], "roll_off_var": [], 
            "roll_off50_mean": [], "roll_off50_var": [],
            "roll_off25_mean": [], "roll_off25_var": [],
            "log_mel_mean": [], "log_mel_var": [], 
            "mfcc_mean": [], "mfcc_var": [], 
            "spec_mean": [], "spec_var": [], 
            "mag_spec_mean": [], "mag_spec_var": [], 
            "mel_mean": [], "mel_var": []
        }

    def amplitude_envelope(self, signal):
        amplitude_envelope = []
        for i in range(0, len(signal), self.HOP_LENGTH):
            current_frame_amplitude_envelope = max(signal[i:i+self.FRAME_SIZE])
            amplitude_envelope.append(current_frame_amplitude_envelope)
        return np.array(amplitude_envelope)
    
    def Rms(self, song):
        return librosa.feature.rms(y = song, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH)

    def Zcr(self, song):
        return librosa.feature.zero_crossing_rate(y = song, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH)

    def Mag_spec(self, song):
        signal_ft = np.fft.fft(song)
        magnitude_spectrum = np.abs(signal_ft)
        return magnitude_spectrum
    
    def spectrogram(self, song):
        song_stft = librosa.stft(song, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE)
        y_song = np.abs(song_stft)**2
        return y_song

    def log_spec(self, song):    
        spec_song = self.spectrogram(song)
        return librosa.power_to_db(spec_song)

    def log_mel(self, song, samp_rate):
        mel_spectrogram = librosa.feature.melspectrogram(y = song, n_fft= 2048, sr = samp_rate, hop_length = 512 ,n_mels=50)
        return librosa.power_to_db(mel_spectrogram)
    def Mfcc(self, song, samp_rate):
        return librosa.feature.mfcc(y = song, n_mfcc=13, sr=samp_rate, hop_length=512)
    def delta_mfcc(self, song, samp_rate):
        mfccs = Mfcc (song, samp_rate)

        return 
    # Define remaining functions here...

    def extract_features(self):
        s = time.time()

        for i in range(len(self.data_paths)):
            sample , sr = librosa.load(self.data_paths[i][0])

            ae = self.amplitude_envelope(sample)
            ae_m, ae_v = ae.mean(), ae.var()
            self.features_list["ae_mean"].append(ae_m)
            self.features_list["ae_var"].append(ae_v)

            rms = self.Rms(sample)
            rms_m, rms_v = rms.mean(), rms.var()
            self.features_list["rms_mean"].append(rms_m)
            self.features_list["rms_var"].append(rms_v)

            # Continue like this for all your features...

        e = time.time()
        print((e - s)/60 , "mins")
        return self.features_list
