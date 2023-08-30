from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkinter.messagebox
import wavio
import threading
import sys
import librosa
import os
import time

class StartScreen:
    def __init__(self, root):
        self.root = root
        self.similarity_values = []
        self.root.title("speech_recognition application_ex1")
        self.button_padding = 10


        self.root.geometry("350x700")  # 여기서 크기를 조절


        self.start_button = tk.Button(self.root, text="시작하기", command=self.start_app)
        self.start_button.pack(pady=self.button_padding, fill=tk.X)

        self.check_learning_button = tk.Button(self.root, text="나의 학습량 확인하기", command=self.check_learning)
        self.check_learning_button.pack(pady=self.button_padding, fill=tk.X)

        self.exit_button = tk.Button(self.root, text="종료하기", command=self.root.quit)
        self.exit_button.pack(pady=self.button_padding, fill=tk.X)

        # 각 행(row)과 열(column)에 동일한 가중치를 부여하여 가운데 정렬을 보장합니다.
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.similarity_values = []

    def start_app(self):
        print("앱을 시작합니다.")
        # Hide StartScreen's widgets
        for widget in self.root.winfo_children():
            widget.pack_forget()
        # Create and show RecorderApp's widgets
        self.recorder_app = RecorderApp(self.root)
        self.average_similarity = None

    def check_learning(self):
        if self.similarity_values:
            print("나의 학습량 확인:")
            for idx, value in enumerate(self.similarity_values, start=1):
                print(f"문제 {idx}: {value:.2f}%")
        else:
            print("아직 학습량 데이터가 없습니다.")

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("speech_recognition application_ex1")

        self.root.geometry("350x700")  # 여기서 크기를 조절

        self.button_padding = 10

        self.tfidf_matrix = None  # Initialize as None

        self.current_question = 0
        self.questions = [
            "저 지금 점심 먹으러 나갈까 하는데 같이 가실래요?",
            "낮에 이야기했던 부분에 대해서 상의해 보아야 할 것 같습니다.",
            "공장을 둘러보시고 나면 저희 제품에 대해 더 많이 알게 되실 겁니다.",
        ]
        self.text_label = tk.Label(self.root, text=f"'{self.questions[self.current_question]}'")
        self.text_label.pack(pady=self.button_padding)


        self.record_button = tk.Button(self.root, text="녹음하기", command=self.start_recording)
        self.record_button.pack(pady=self.button_padding)

        self.stop_record_button = tk.Button(self.root, text="녹음 끝내기", state=tk.DISABLED, command=self.stop_recording)
        self.stop_record_button.pack(pady=self.button_padding)

        self.check_pronunciation_button = tk.Button(self.root, text="나의 발음 확인하기", state=tk.DISABLED, command=self.check_pronunciation)
        self.check_pronunciation_button.pack(pady=self.button_padding)

        self.result_text = tk.Text(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.tag_configure("mismatch", foreground="red")

        self.back_button = tk.Button(self.root, text="돌아가기", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=self.button_padding, fill=tk.X)

        self.next_button = tk.Button(self.root, text="다음 문제로 가기", command=self.open_ex2_window)
        self.next_button.pack(pady=self.button_padding)


        self.exit_button = tk.Button(self.root, text="종료하기", command=self.exit_program)
        self.exit_button.pack(pady=self.button_padding)


     
        self.status_label = tk.Label(self.root, text="", fg="blue")
        self.status_label.pack()

        self.is_recording = False
        self.recording_data = []

        self.record_button.pack(side=tk.TOP, padx=self.button_padding, fill=tk.X)
        self.stop_record_button.pack(side=tk.TOP, padx=self.button_padding, fill=tk.X)
        self.check_pronunciation_button.pack(side=tk.TOP, padx=self.button_padding, fill=tk.X)
        self.back_button.pack(side=tk.TOP, padx=self.button_padding, fill=tk.X)
        self.next_button.pack(side=tk.TOP, padx=self.button_padding, fill=tk.X)
        #self.exit_button.pack(side=tk.LEFT, padx=self.button_padding, fill=tk.X)

        self.similarity_scores = []  # List to store similarity scores for each question

    def start_recording(self):
        self.record_button.config(state=tk.DISABLED)
        self.stop_record_button.config(state=tk.NORMAL)
        self.check_pronunciation_button.config(state=tk.DISABLED)
        self.exit_button.config(state=tk.DISABLED)
        self.status_label.config(text="녹음 중...")
        self.is_recording = True
        self.recording_data = []

        self.record_thread = threading.Thread(target=self.record)
        self.record_thread.start()

    def stop_recording(self):
        self.record_button.config(state=tk.NORMAL)
        self.stop_record_button.config(state=tk.DISABLED)
        self.check_pronunciation_button.config(state=tk.NORMAL)
        self.exit_button.config(state=tk.NORMAL)
        self.status_label.config(text="")
        self.is_recording = False

        # 녹음 파일 저장
        if self.recording_data:
            # 고유한 파일 이름 생성
            timestamp = time.strftime("%Y%m%d%H%M%S")
            filename = f"recording_{timestamp}.wav"
            wav_filename = os.path.join("recordings", filename)  # recordings 폴더 안에 저장

            # 녹음 데이터 저장
            wavio.write(wav_filename, np.array(self.recording_data), 44100, sampwidth=2)

            # .wav 파일을 .txt 파일로 변환
            txt_filename = os.path.join("recordings", f"recording_{timestamp}.txt")
            self.wav_to_txt(wav_filename, txt_filename)

            # 발음 확인 실행
            self.run_pronunciation_check(txt_filename)

    def go_back(self):
        if self.current_question > 0:
            self.current_question -= 1
            new_question = self.questions[self.current_question]
            self.text_label.config(text=f"'{new_question}'")
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            self.status_label.config(text="이전 문제입니다.", fg="blue")

    def wav_to_txt(self, wav_filename, txt_filename):
        # .wav 파일을 .txt 파일로 변환
        r = sr.Recognizer()
        korean_wav, rate = librosa.load(wav_filename, sr=None)
        korean_audio = sr.AudioData(korean_wav.tobytes(), korean_wav.size * korean_wav.itemsize, rate)

        try:
            result = r.recognize_google(korean_audio, language='ko-KR')
            # .txt 파일을 해당 디렉토리에 저장
            with open(self.txt_filepath, 'w', encoding='utf-8') as f:
                f.write(result)
        except sr.UnknownValueError:
            print("음성을 인식할 수 없습니다.")
        except sr.RequestError:
            print("음성 인식 서비스에 접근할 수 없습니다.")

    def record(self):
        fs = 44100

        while self.is_recording:
            chunk = sd.rec(fs, samplerate=fs, channels=2)
            sd.wait()
            self.recording_data.extend(chunk)

    def check_pronunciation(self):
        if self.recording_data:
            self.status_label.config(text="발음 확인 중...")
            self.check_pronunciation_button.config(state=tk.DISABLED)
            
            threading.Thread(target=self.run_pronunciation_check).start()

    def run_pronunciation_check(self):
        # 파일에서 문장 읽어오기
        def read_sentence_from_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = f.readlines()
            return [sentence.strip() for sentence in sentences]

        # 파일 경로
        #file_path1 = self.txt_filename  #들어오는 외국인 발음 텍스트파일
        file_path1 = 'D:/speech_recognition/wavefile_data/text/foreigner_result.txt'
        file_path2 = 'D:/speech_recognition/wavefile_data/text/train_man_02_result.txt' #한국인 정답 발음

        # 파일에서 문장 읽어오기
        sentences1 = read_sentence_from_file(file_path1)
        sentences2 = read_sentence_from_file(file_path2)

        # 문장 비교
        all_sentences = sentences1 + sentences2
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_sentences)

        # 유사도 계산 및 출력
        result_text = "발음 분석 결과:\n"
        similarity_percentages = []  # List to store similarity percentages
        for i in range(len(sentences1)):
            for j in range(len(sentences2)):
                similarity_score = cosine_similarity(tfidf_matrix[i], tfidf_matrix[len(sentences1) + j])
                similarity_percentage = similarity_score[0][0] * 100
                self.similarity_scores.append(similarity_percentage)
                sentence1 = sentences1[i]
                sentence2 = sentences2[j]

                result_text += f"사용자 발음 문장: {sentence1}\n"
                result_text += f"한국인 발음 문장: {sentence2}\n"
                result_text += f"나의 발음은 얼마나 비슷할까?: {similarity_percentage:.2f}%\n"

            if similarity_percentage < 90:
                result_text += f"'{sentence1.split(' ')[-1]}' 이후로 정확하지 않습니다.\n"
                self.result_text.tag_add("mismatch", tk.END + "-1l", tk.END)  # 빨간색 서식 적용
                
            result_text += "=" * 20 + "\n"

        # 텍스트 위젯에 결과 출력
        self.result_text.config(state=tk.NORMAL)  # 편집 가능한 상태로 설정
        self.result_text.delete(1.0, tk.END)  # 모든 텍스트 삭제
        self.result_text.insert(tk.END, result_text)  # 새로운 결과 입력
        self.result_text.config(state=tk.DISABLED)  # 편집 불가능한 상태로 설정
        self.status_label.config(text="발음 확인 완료")
        self.check_pronunciation_button.config(state=tk.NORMAL)

                # if similarity_percentage < 50:
                #     self.result_text.insert(tk.END, f"'문장 1'에서 '{sentence1.split(' ')[-1]}' 이후로 정확하지 않습니다.\n", "mismatch")
                # result_text += "=" * 20 + "\n"
        self.tfidf_matrix = tfidf_matrix
                    

        return similarity_percentages

        
    def open_ex2_window(self):
        # 텍스트 위젯의 내용과 상태 레이블을 초기화합니다.
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.status_label.config(text="다음 문제입니다.", fg="blue")

        # 다음 문제로 이동합니다.
        self.current_question += 1
        if self.current_question < len(self.questions):
            new_question = self.questions[self.current_question]
            self.text_label.config(text=f"'{new_question}'")
        # Check if all questions are done
        if self.current_question == len(self.questions):
            average_score = sum(self.similarity_scores) / len(self.similarity_scores)
            self.status_label.config(text="모든 문제를 완료했습니다.", fg="green")
            tkinter.messagebox.showinfo("전체 평균 점수", f"모든 문제를 완료했습니다.전체 평균 점수는 {average_score:.2f}%입니다.")
            self.calculate_average_similarity()  # 평균 유사도 계산
            self.display_average_similarity_popup()  # 평균 유사도 팝업 표시
            self.reset_app()  # 팝업 확인 시 앱 초기화

    def reset_app(self):
        self.current_question = 0
        self.text_label.config(text=f"'{self.questions[self.current_question]}'")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.status_label.config(text="")
        self.average_similarity = None
        self.tfidf_matrix = None

        self.record_button.config(state=tk.NORMAL)
        self.stop_record_button.config(state=tk.DISABLED)
        self.check_pronunciation_button.config(state=tk.DISABLED)
        self.exit_button.config(state=tk.NORMAL)

    def calculate_average_similarity(self):
        if self.tfidf_matrix is None:
            return 0

        num_questions = len(self.questions)
        total_similarity_percentage = 0
        total_comparisons = 0

        num_tfidf_rows, _ = self.tfidf_matrix.shape  # Get the number of rows in the matrix

        for i in range(num_questions):
            for j in range(num_questions):
                if i < num_tfidf_rows and j < num_tfidf_rows:
                    similarity_score = cosine_similarity(
                        self.tfidf_matrix[i], self.tfidf_matrix[j]
                    )
                    similarity_percentage = similarity_score[0][0] * 100
                    total_similarity_percentage += similarity_percentage
                    total_comparisons += 1

        average_similarity = total_similarity_percentage / total_comparisons
        return average_similarity



    def display_average_similarity_popup(self):
        if self.average_similarity is not None:
            popup_message = f"전체 문제의 평균 유사도: {self.average_similarity:.2f}%"
            popup_result = tkinter.messagebox.showinfo("평균 유사도", popup_message)
            if popup_result == "ok":
                self.reset_app()  # 팝업 확인 시 앱 초기화

    def exit_program(self):
        if self.tfidf_matrix is not None:
            average_similarity_percentage = self.calculate_average_similarity()

            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(
                tk.END, f"전체 문제의 평균 유사도: {average_similarity_percentage:.2f}%\n\n"
            )
            self.result_text.config(state=tk.DISABLED)

        sys.exit()




if __name__ == "__main__":
    root = tk.Tk()
    start_screen = StartScreen(root)
    root.mainloop()