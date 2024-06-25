from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
from queue import Queue
import llama_run_model
import fitz
import os
import json


#import configparser
#config = configparser.ConfigParser()
#config.read('ngrok_key.ini')

import secrets
secret_key = secrets.token_hex(16)
print("비밀키 생성: ", secret_key)


chatbot1 = llama_run_model.model() #챗봇 모델 생성


app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
socketio = SocketIO(app, cors_allowed_origins="*")

# 메시지 대기열
message_queue = Queue()

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER): 
    os.makedirs(UPLOAD_FOLDER) # 파일 받아 저장할 폴더 생성

@app.route('/')

def chatbot(message=""):
    response = chatbot1.llama(message[1])
    time.sleep(1)
    return response

def message_processor():
    while True:
        message, sid = message_queue.get()
        print("요청 처리 시작: ",message[0],", 파일: ",message[1])
        extracted_text=""
        
        
        #주관식 문제 생성 요청일 경우
        if message[0] == "make_q":                 
            extracted_text = get_txt(message[1], sid, message[0]) #텍스트 추출
               
            response_q = chatbot1.llama_make_question(extracted_text)  #추출된 텍스트를 문제생성 봇으로 전달/ 답변 받음
            response_q = json.loads(response_q, strict=False) # json타입(딕셔너리)으로 변환
            
            try_count = 0
            while try_count < 3:  #답변이 제대로 생성이 안될 경우 3회 시도
                if "" in response_q.values():
                    print(f"답변을 재생성 합니다. 시도횟수: {try_count+1}")
                    response_q = chatbot1.llama_make_question(extracted_text)
                    response_q = json.loads(response_q, strict=False)
                    try_count += 1
                else: break
            
            if "" in response_q.values():
                response_q["question"] = response_q["correct_answer"] = "<생성 실패>"
                
                
            print("답변타입:",type(response_q), "전송될데이터:", response_q)
            socketio.emit('make_q_result', response_q, to=sid)
            message_queue.task_done()
            
            # 처리 끝날때 받은 파일 제거
            if sid in os.listdir(f"./uploads/{message[0]}"): 
                for i in os.listdir(f"./uploads/{message[0]}/{sid}"):
                    os.remove(f"./uploads/{message[0]}/{sid}/{i}")
                os.rmdir(f"./uploads/{message[0]}/{sid}")
                print("삭제됨: ",message[0]," sid = ",sid)
        
        
        #객관식 문제생성 요청
        elif message[0] == "make_q_2":              
            
            extracted_text = get_txt(message[1],sid, message[0])
            response_q = chatbot1.llama_make_question_3(extracted_text)  #추출된 텍스트를 문제생성 봇으로 전달/ 답변 받음
            print(response_q)
            response_q = json.loads(response_q, strict=False) # json타입(딕셔너리)으로 변환
            
            try_count = 0
            while try_count < 3:  #답변이 제대로 생성이 안될 경우 3회 시도
                if response_q["correct_answer"] not in ["A", "B", "C", "D"] or "" in response_q.values():
                    print(f"답변을 재생성 합니다. 시도횟수: {try_count+1}")
                    response_q = chatbot1.llama_make_question_3(extracted_text)
                    response_q = json.loads(response_q, strict=False)
                    try_count += 1
                else: break
                
            if try_count == 3:
                for j in response_q.keys():
                    response_q[j] = "<생성 실패>"

                
            print("답변타입:",type(response_q), "전송될데이터:", response_q)
            socketio.emit('make_q_2_result', response_q, to=sid)
            message_queue.task_done()
                    
            if sid in os.listdir(f"./uploads/{message[0]}"): 
                for i in os.listdir(f"./uploads/{message[0]}/{sid}"):
                    os.remove(f"./uploads/{message[0]}/{sid}/{i}")
                os.rmdir(f"./uploads/{message[0]}/{sid}")
                print("삭제됨: ",message[0]," sid = ",sid)
                    
        #최초 요약 채팅
        elif message[0] == "make_summary":
            extracted_text = get_txt(message[1],sid, message[0])
        
            response = chatbot1.llama_summary_chat_first(extracted_text, uid = message[2])
            print("답변타입:",type(response), "전송될데이터:", response)
            socketio.emit('summary_result', response, to=sid)
            message_queue.task_done()
                    
        #최초이후 요약 관련 질의응답
        elif message[0] == "summary_chat":
            
            response = chatbot1.llama_summary_chat(message[1],uid = message[2])
            
            socketio.emit('summary_result', response, to=sid)
            message_queue.task_done()
            
        elif message[0] == "daily_chat": #일상 대화
            response = chatbot1.llama_daily_chat(message[1],uid = message[2])
            socketio.emit('daily_response', {"message":str(response), "uid":message[2]}, to=sid)
            message_queue.task_done()
            
        elif message[0] == "analyze_emotion": #감성분석
            exist_log_list = os.listdir('./daily_chatlog')
            if exist_log_list:
                response_list = {"result":[]}
                for i in exist_log_list:
                    
                    result_temp = chatbot1.llama_analyze_emotion(i)
                    result_temp = json.loads(result_temp)
                    result_temp["uid"] = i
                    response_list["result"].append(result_temp)
            else: 
                response_list["result"].append({"conclusion":"","emotion":"로그 비었음","uid":""})
            
            print("답변타입:",type(response_list), "전송될데이터:", response_list)
            socketio.emit('emotion_analyze_result', response_list, to=sid)
            message_queue.task_done()
            
        else:
            processed_message = chatbot(message)
            socketio.emit('bot_response', processed_message, to=sid)
            message_queue.task_done()
            
            
            
#======================================================================================================#
#(부가기능 모음)

def get_txt(filename,sid,request_type):  
    
    if filename.split(".")[1] == "pdf": #pdf파일에서 텍스트 추출
        # Open a PDF file
        pdf_document = filename
        doc = fitz.open(f'./uploads/{request_type}/{sid}/{pdf_document}')

        # Initialize an empty string to store extracted text
        extracted_text = ""

        # Iterate through each page and extract text
        for page_num in range(doc.page_count):
            page = doc[page_num]
            extracted_text += page.get_text()

        # Close the PDF document
        doc.close()
    elif filename.split(".")[1] == "txt":    #txt인 경우 텍스트 추출
        with open(f"./uploads/{request_type}/{sid}/{filename}","r", encoding="utf-8") as f:
            extracted_text = f.read()
                    
    extracted_text=extracted_text.replace("\n","") #줄바꿈 제거
    return extracted_text

def save_file(filename, file_data, sid,request_type): #파일 저장 함수
    
    personal_path = f'{UPLOAD_FOLDER}/{request_type}/{sid}'
    
    if not os.path.exists(personal_path): 
        os.makedirs(personal_path) # 파일 받아 저장할 폴더 생성
        
    file_path = os.path.join(personal_path, filename)
    with open(file_path, 'wb') as f:
        f.write(bytearray(file_data))
        
#########################################################################################################
#########################################################################################################

@socketio.on('message')
def handle_message(message):
    sid = request.sid
    chat_message = ["chat",message]
    message_queue.put((chat_message, sid))

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('make_summary') #최초 파일 요약 채팅 요청
def make_summary(data):
    sid = request.sid
    

    filename = data['filename']
    file_data = data['data']
    uid = data['uid']
    message =["make_summary",filename, uid]
    save_file(filename, file_data, sid, message[0])
    message_queue.put((message, sid))
    
    print("요약생성 요청받음. 파일:", filename)
    
@socketio.on('summary_chat') #최초 이후 관련 채팅
def summary_chat(data):
    sid = request.sid
    
    uid = str(data['uid'])
    chat_message = ["summary_chat",data['message'],uid]
    message_queue.put((chat_message, sid))
    
@socketio.on('make_q')
def make_question(data):
    sid = request.sid

    filename = data['filename']
    file_data = data['data']
    message =["make_q",filename]
    
    # 파일 저장
    # file_path = os.path.join(UPLOAD_FOLDER, filename)
    # with open(file_path, 'wb') as f:
    #     f.write(bytearray(file_data))
    save_file(filename, file_data, sid, message[0])

    emit('upload_response', {'message': f'파일: {filename}  업로드 성공!'})
    message_queue.put((message, sid))
    print("문제생성 요청받음. 파일:", filename)
    
@socketio.on('make_q_2')
def make_question(data):
    sid = request.sid

    filename = data['filename']
    file_data = data['data']
    message =["make_q_2",filename]
    
    # 파일 저장
    # file_path = os.path.join(UPLOAD_FOLDER, filename)
    # with open(file_path, 'wb') as f:
    #     f.write(bytearray(file_data))
    save_file(filename, file_data, sid, message[0])

    emit('upload_response', {'message': f'파일: {filename}  업로드 성공!'})
    message_queue.put((message, sid))
    print("문제생성 요청받음. 파일:", filename)
    
@socketio.on('daily_chat')  #일상 대화 챗봇
def daily_chatbot(data):
    sid = request.sid
    uid = data['uid']
    chat_message = ["daily_chat",data['message'],uid]
    message_queue.put((chat_message, sid))
    
    
@socketio.on('analyze_emotion')  #감정분석 요청
def emotion(data):
    sid = request.sid
    chatlog = ["analyze_emotion", data['uid']]
    message_queue.put((chatlog,sid))









if __name__ == '__main__':
    # 메시지 처리 스레드 시작
    processor_thread = threading.Thread(target=message_processor, daemon=True)
    processor_thread.start()

    socketio.run(app, port=5050)
    

# @app.route('/')
# def index():
#     return "Server is running"

# @socketio.on('message')
# def handle_message(msg):
#     print(f"Message: {msg}")
#     response = chatbot(msg)
#     send(response, broadcast=True)

# if __name__ == '__main__':
#     socketio.run(app, debug=True, port=5050)