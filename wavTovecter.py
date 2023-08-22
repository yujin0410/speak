import speech_recognition as sr
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import os

#korean_wavfile_path = './wavefile_data/foreigner.wav'
korean_wavfile_path = './wavefile_data/train_man_01.wav'

file_name_without_extension = os.path.splitext(os.path.basename(korean_wavfile_path))[0]

r = sr.Recognizer()

plt.style.use('ggplot')

korean_wav, rate = librosa.load(korean_wavfile_path, sr=None)

plt.figure(figsize=(14, 4))
plt.plot(korean_wav)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

sd.play(korean_wav, rate)
sd.wait()

korean_audio = sr.AudioFile(korean_wavfile_path)

with korean_audio as source:
    audio = r.record(source)

try:
    result = r.recognize_google(audio, language='ko-KR')
    print("음성 인식 결과:", result)

    # 텍스트 파일에 결과 저장
    output_folder = './wavefile_data/text/'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, file_name_without_extension + '_result.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)

    print("음성 인식 결과가 '{}' 파일에 저장되었습니다.".format(output_file))
except sr.UnknownValueError:
    print("음성을 인식할 수 없습니다.")
except sr.RequestError:
    print("음성 인식 서비스에 접근할 수 없습니다.")