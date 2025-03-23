# 1. Base Image 설정 (Python 기반 환경 사용)
FROM python:3.8-slim

# 시스템 패키지 업데이트 및 pip, setuptools 최신 버전으로 업데이트
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    && pip install --upgrade pip setuptools

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 파일 복사 (코드 및 모델)
COPY app.py /app
COPY kobert_emotion.onnx /app

# 4. 종속 라이브러리 설치
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Flask 서버 실행
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]