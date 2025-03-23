import requests

# Flask 서버 URL
flask_url = "http://127.0.0.1:5000/analyze"

# 백엔드 서버 URL
backend_url = "http://127.0.0.1:8000/receive_sentiment"

# 사용자 입력 데이터
user_input = {"text": "오늘 하루는 정말 행복했어요!"}

# Flask 서버로 감정 분석 요청
response = requests.post(flask_url, json=user_input)

if response.status_code == 200:
    # 감정 분석 결과
    sentiment_result = response.json()
    print("Sentiment Analysis Result:", sentiment_result)

    # 백엔드 서버로 결과 전송
    backend_response = requests.post(backend_url, json=sentiment_result)
    if backend_response.status_code == 200:
        print("Result sent to backend successfully!")
    else:
        print("Failed to send result to backend:", backend_response.text)
else:
    print("Sentiment Analysis Failed:", response.text)