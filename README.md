# chatupschool_chatbot
챗업스쿨용 챗봇 구동 앱  

## 챗업스쿨 프로젝트 깃허브
https://github.com/Cuunlang/chat-school  

## 챗봇 구동을 위한 앱 설계
- api제작을 위해 Flask, socketio를 사용
- 외부 통신을 위한 터널링 프로그램 **ngrok**사용
- model폴더에 **gguf**형식의 모델을 넣고 해당 파일 명으로 경로를 지정 후 실행

## 챗봇 실행 하기
1. (선택) 외부 접근 가능한 포트로 url 생성  
   > 1-1. Ngrok 가입및 인증토큰 발급  
   > 1-2. IDE상 터미널 실행 (필자는 conda가상환경에서 실행)  
   > 1-3. Ngrok 설치 및 토큰 입력  
   pip install ngrok  
   ngrok config add-authtoken $YOUR_AUTHTOKEN  
   > 1-4. Fowarding url 복사 및 확인  
2. model에 finetune한 gguf모델 넣기  
3. llama_run_model.py 에서 gguf파일 이름으로 경로지정
4. 새 터미널에서 앱 실행  
    python app.py
