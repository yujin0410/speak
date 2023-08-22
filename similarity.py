from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 파일에서 문장 읽어오기
def read_sentence_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return [sentence.strip() for sentence in sentences]

# 파일 경로
file_path1 = 'D:/speech_recognition/wavefile_data/text/foreigner_result.txt'
file_path2 = 'D:/speech_recognition/wavefile_data/text/train_man_02_result.txt'

# 파일에서 문장 읽어오기
sentences1 = read_sentence_from_file(file_path1)
sentences2 = read_sentence_from_file(file_path2)

# 문장 비교
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences1 + sentences2)
# 유사도 계산 및 출력
print("문장 간 유사도:")
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        similarity_score = cosine_similarity(tfidf_matrix[i], tfidf_matrix[len(sentences1) + j])
        print(f"문장 1: {sentences1[i]}")
        print(f"문장 2: {sentences2[j]}")
        print("유사도:", similarity_score[0][0])
        if similarity_score[0][0] < 0.5:
            print(f"'문장 1'에서 '{sentences1[i].split(' ')[-1]}' 이후로 정확하지 않습니다.")
        print("=" * 20)
