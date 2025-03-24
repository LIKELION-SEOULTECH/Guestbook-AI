import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import numpy as np
import onnxruntime as ort

# KoBERT 토크나이저
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

# Flask 서버
app = Flask(__name__)

# BERT 분류기 모델 로드
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, segment_ids, attention_mask):
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        if self.dr_rate:
            pooler = self.dropout(pooler)
        return self.classifier(pooler)

# 예측 함수
# ONNX 모델 로드
onnx_session = ort.InferenceSession("kobert_emotion.onnx", providers=['CPUExecutionProvider'])


# for node in onnx_session.get_inputs():
#     print("Input name:", node.name, "Shape:", node.shape)

# sentence = "슬픔이 가득한 문장입니다."
# inputs = tokenizer.encode_plus(sentence, max_length=64, padding='max_length', truncation=True, return_tensors="np")
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
# print(f"Original Sentence: {sentence}")
# print(f"Tokenized Sentence: {tokens}")


def predict(sentence):
    # KoBERT 입력값 생성
    inputs = tokenizer.encode_plus(sentence, max_length=64, padding='max_length', truncation=True, return_tensors="np")
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    segment_ids = inputs["token_type_ids"]  # KoBERT에서는 token_type_ids를 segment_ids로 사용
    
    # KoBERT의 [PAD] 토큰 ID 가져오기
    pad_token_id = tokenizer.pad_token_id
    valid_length = (input_ids != pad_token_id).sum(axis=1).tolist()[0]
    # print("Valid Length : ", valid_length)

    # ONNX Runtime 입력값 (이름 변경)
    ort_inputs = {
        "token_ids": input_ids.astype(np.int64),      # ✅ 이름 변경
        "valid_length": np.array([valid_length], dtype=np.int64),  # ✅ 추가
        "segment_ids": segment_ids.astype(np.int64)   # ✅ 이름 변경
    }
    # print("input_ids:", input_ids)
    # print("segment_ids:", segment_ids)
    # print("valid_length:", valid_length)


    # ONNX 추론
    ort_outs = onnx_session.run(None, ort_inputs)
    logits = ort_outs[0]

    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    # print("Softmax probabilities:", probs)  


    # 감정 라벨 매핑
    emotions = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]
    predicted_class = np.argmax(logits, axis=1)[0]
    predicted_emotion = emotions[predicted_class]

    return predicted_emotion, logits[0]



# Flask 라우팅
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    sentence = data['sentence']
    
    predicted_emotion, probs = predict(sentence)
    
    return jsonify({
        'emotion': predicted_emotion,
        'probabilities': probs.tolist()
    })

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)
