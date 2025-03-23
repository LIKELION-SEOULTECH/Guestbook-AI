import torch
import onnx
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import onnxruntime as ort
import os
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer

# Flask 서버 설정
app = Flask(__name__)

# ONNX 모델 로드
onnx_path = '/Users/jogeonhui/Documents/LikeLion/1~2week_development_session/emotion_model.onnx'
ort_session = ort.InferenceSession(onnx_path)

# KoBERT 토크나이저 로드
# tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# 감정 클래스 레이블
emotion_labels = ["불안", "분노", "상처", "슬픔", "당황", "기쁨"]

# 전처리 함수
def preprocess(sentence):
    inputs = tokenizer(
        sentence,
        padding="max_length",
        max_length=10,
        truncation=True,
        return_tensors="np"
    )

    input_ids = np.clip(inputs["input_ids"], 0, tokenizer.vocab_size - 1)
    attention_mask = inputs["attention_mask"]
    token_type_ids = np.zeros_like(input_ids)  # segment_ids 생성
    valid_length = (input_ids != 0).sum(axis=1).tolist()[0]

    return {
        "input": input_ids.astype(np.int64),  # [1, 10]
        "valid_length": np.array([valid_length], dtype=np.int64),  # [1]
        "onnx::Cast_2": token_type_ids.astype(np.int64)  # 💡 여기가 핵심!
    }

# 감정 추론 함수
def predict_emotion(sentence):
    onnx_inputs = preprocess(sentence)
    ort_outs = ort_session.run(None, onnx_inputs)
    predicted_prob = ort_outs[0]
    predicted_label = np.argmax(predicted_prob, axis=1)[0]
    return predicted_label, predicted_prob

# API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sentence = data.get("sentence", "")
        
        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400
        
        predicted_label, predicted_prob = predict_emotion(sentence)
        
        return jsonify({
            "predicted_label": emotion_labels[predicted_label],
            "predicted_probabilities": predicted_prob.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
