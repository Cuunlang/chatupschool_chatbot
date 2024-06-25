from llama_cpp import Llama
import os
import json
#from transformers import AutoTokenizer, PreTrainedTokenizerFast

model_8bit_maal="./model/ggml-llama3-MAAL-Q8_0-v2.gguf"
model_new = "./model/model-unsloth.Q8_0.gguf"
model_new2 = "./model/ggml-llama3-alpha-ko-instruct-q8_0.gguf"
model_new3 = "./model/model-llama3-ko-alpha-finetuned-Q8_0.gguf"
model_new4 = "./model/ggml-llama3-alpha-ko-emotion-q8_0.gguf"
model_new5 = "./model/ggml-llama3-genQuiz-q8_0.gguf"
#tokenizer1 = AutoTokenizer.from_pretrained('allganize/Llama-3-Alpha-Ko-8B-Instruct', legacy=True)

class model:
    def __init__ (self):
        self.llm = Llama(
            model_path = model_new2,
            chat_format= "llama-3",
            n_gpu_layers = -1,
            n_ctx=40240, # context window 입력 토큰 크기 조절 (기본값 512)
            )
        self.messages_hist = [
            {
            "role": "system",
            "content": "너는 학생을 가르치는 교사다. 처음 입력받은 내용을 간단하게 요약하여 답변하고, 이후 질문을 받으면 처음 유저에게 입력받은 내용을 기반으로 대답하라."
            }
        ]

    def llama(self, message):
        print("입력된 질문: ",message)
        self.messages_hist.append(
                {"role": "user",
                "content": f"{message}"})

        result = self.llm.create_chat_completion(
            max_tokens=512,
            frequency_penalty= 1,
            repeat_penalty= 1.1,
            temperature= 0.8,
            messages = [{
                            "role": "system",
                            "content": "너는 학생을 가르치는 교사다. 유저에게 입력받은 내용을 기반으로 친절하게 대답하라. 모르는 내용은 절대로 모른다고 답해라."
                        },
                        {
                            "role": "user",
                            "content": f"{message}"
                        }
                ]
            )
        self.messages_hist.append({
            "role": "assistant",
            "content": result['choices'][0]['message']['content']
            })
        print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
        return result['choices'][0]['message']['content']
    
    def llama_make_question(self,extracted_text):
        result = self.llm.create_chat_completion(
            max_tokens=512,
            frequency_penalty= 1,
            repeat_penalty= 1.1,
            temperature= 0.8,
            messages = [
                {
                    "role": "system",
                    "content": '''당신은 퀴즈생성 전문가입니다. 입력받은 내용을 기반으로 학생을 교육하기위한 주관식 문제를 만들어서 JSON으로 출력합니다. 출력형식은 다음과 같습니다.
                    
                    - 'question': 퀴즈의 질문을 입력합니다,
                    - 'correct_answer': 퀴즈의 질문에 대한 정답을 입력합니다''',
                },
                {"role": "user", "content": extracted_text},
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}, "correct_answer":{"type":"string"}},
                    "required": ["question", "correct_answer"],
                }}
                    )
        print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
        return result['choices'][0]['message']['content']
    
    # def llama_make_question_2(self, extracted_text):
    #     result = self.llm.create_chat_completion(
    #         max_tokens=1024,
    #         frequency_penalty= 1,
    #         repeat_penalty= 1.1,
    #         temperature= 0.8,
    #         messages = [
    #             {
    #                 "role": "system",
    #                 "content": "This is a useful helper for you to output as Json. Based on the information entered by the user, a four-choice multiple-choice question must be created. You must enter the question in question in 'question', the question in question in '1' to '4', and the number of the correct answer in 'correct_answer'. 당신은 Json으로 출력하는 유용한 도우미 입니다. user에게 입력받은 내용을 기반으로 4지선다형 객관식 문제를 만들어야 합니다. 'question'에는 문제의 질문을, '1'부터 '4'에는 문제의 문항을, 'correct_answer'에는 정답의 번호를 넣어야 합니다. ",
    #             },
    #             {"role": "user", "content": f"{extracted_text}"},
    #         ],
    #         response_format={
    #             "type": "json_object",
    #             "schema": {
    #                 "type": "object",
    #                 "properties": {"question": {"type": "string"},
    #                                "1":{"type": "string"},"2":{"type": "string"},"3":{"type": "string"},"4":{"type": "string"},
    #                                "correct_answer":{"type":"string"}},
    #                 "required": ["question", "correct_answer", "1", "2", "3", "4"],
    #             }}
    #                 )
    #     print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
    #     return result['choices'][0]['message']['content']
    
    
    #- "A", "B", "C", "D": 질문에 대한 잘못된 답 3개와 올바른 답 하나를 생성하여 "A", "B", "C", "D"에 각각 랜덤으로 입력합니다.
    def llama_make_question_3(self, extracted_text):
        result = self.llm.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": '''당신은 퀴즈 생성 전문가입니다. 입력받은 내용을 기반으로 4개의 보기를 가지는 퀴즈를 생성하고, JSON 형식으로 출력하는 역할을 수행합니다. 출력 형식은 다음과 같습니다:

                    - "question": 제공된 user 내용에 대한 퀴즈의 질문 입력합니다.
                    - "A", "B", "C", "D": 질문에 대한 보기 4개를 "A", "B", "C", "D"에 각각 입력합니다.
                    - "correct_answer": 보기 중 올바른 답의 알파벳 하나만 입력합니다.

                    ''',
                },
                {"role": "user", "content": f"{extracted_text}"},
            ],
            max_tokens=1024,
            frequency_penalty= 1,
            repeat_penalty= 1.1,
            temperature= 0.8,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"question": {"type": "string"},
                                   "A":{"type": "string"},"B":{"type": "string"},"C":{"type": "string"},"D":{"type": "string"},
                                   "correct_answer":{"type":"string"}},
                    "required": ["question", "A", "B", "C", "D", "correct_answer"],
                }}
                    )
        print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
        return result['choices'][0]['message']['content']
    
    # def llama_make_question_3_multi(self, extracted_text):
    #     result = self.llm.create_chat_completion(
    #         messages = [
    #             {
    #                 "role": "system",
    #                 "content": '''당신은 퀴즈 생성 전문가입니다. user의 입력을 기반으로 4개의 보기를 가지는 퀴즈를 생성하고, JSON 스키마로 출력하는 역할을 수행합니다. 출력 형식은 다음과 같습니다:

    #                 - "Q1", "Q2", "Q3"는 각각 하나의 퀴즈를 나타냅니다.
    #                 - 각각의 퀴즈에서 "question"에는 퀴즈의 질문을, "A", "B", "C", "D"에는 그 문제의 보기를 가지며 올바른 보기는 하나입니다.
    #                 - 각각의 퀴즈에서 "correct_answer"에는 올바른 정답인 보기를 입력합니다 (예: "A", "B", "C", "D").


    #                 이제 사용자가 제공한 컨텐츠를 입력받아 위의 형식에 맞춰 여러 개의 4지 선다형 문제를 생성하세요. 각 문제의 보기는 해당 질문의 답변으로 의미 있는 선택지를 포함하세요.

    #                 ''',
    #             },
    #             {"role": "user", "content": f"{extracted_text}"},
    #         ],
    #         max_tokens=1024,
    #         frequency_penalty= 1,
    #         repeat_penalty= 1.1,
    #         temperature= 0.8,
    #         response_format={
    #             "type": "json_object",
    #             "schema": {
    #                 "type": "object",
    #                 "properties": {"Q1":{"question": {"type": "string"},
    #                                "A":{"type": "string"},"B":{"type": "string"},"C":{"type": "string"},"D":{"type": "string"},
    #                                "correct_answer":{"type":"string"}},
    #                                "Q2":{"question": {"type": "string"},
    #                                "A":{"type": "string"},"B":{"type": "string"},"C":{"type": "string"},"D":{"type": "string"},
    #                                "correct_answer":{"type":"string"}},
    #                                "Q3":{"question": {"type": "string"},
    #                                "A":{"type": "string"},"B":{"type": "string"},"C":{"type": "string"},"D":{"type": "string"},
    #                                "correct_answer":{"type":"string"}}
    #                                },
    #                 "required": ["question", "A", "B", "C", "D", "correct_answer"],
    #             }}
    #                 )
    #     print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
    #     return result['choices'][0]['message']['content']
    
    
    
    
    def llama_summary_chat_first(self, message, uid):
        if uid not in os.listdir('./chat_log'): #처음 시도일 경우
            os.makedirs(f'./chat_log/{uid}')   #uid로 경로 생성
        else:
            for i in os.listdir(f'./chat_log/{uid}'): #이미 있을경우 기존 로그 제거
                os.remove(f'./chat_log/{i}')
            
        newchat ={
            "uid": uid,
            "chat":[
                {
                    "role": "system",
                    "content": "너는 학생을 가르치는 교사다. 유저에게 입력받은 내용을 기반으로 친절하게 대답하라. 모르는 내용은 절대로 모른다고 답해라."
                },
                {
                    "role": "user",
                    "content": f"다음 내용을 요약해줘. {message}"
                }
                ]
            }
        
        #파일 내용을 요약 요청하는 프롬프트를 Json으로 저장
        with open(f'./chat_log/{uid}/log.json', 'w', encoding='utf-8') as f:
            json.dump(newchat, f, ensure_ascii=False, indent="\t") 
            
        
        #print("입력된 질문: ",message)
        # self.messages_hist.append(
        #         {"role": "user",
        #         "content": f"{message}"})

        result = self.llm.create_chat_completion(
            max_tokens=512,
            frequency_penalty= 1,
            repeat_penalty= 1.1,
            temperature= 0.8,
            messages = list(newchat["chat"])
            )
        
        #생성된 답변 추가
        newchat["chat"].append({
            "role": "assistant",
            "content": result['choices'][0]['message']['content']
            })
        
        with open(f'./chat_log/{uid}/log.json', 'w', encoding='utf-8') as f:
            json.dump(newchat, f, ensure_ascii=False, indent="\t") 
        
        
        print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
        return result['choices'][0]['message']['content']
    
    #요약채팅 두번쨰부터
    def llama_summary_chat(self, message, uid):
        if uid not in os.listdir('./chat_log'): #처음 시도일 경우
            return "파일이 제대로 입력되지 않았거나 로그가 없습니다."
        else:
            with open(f'./chat_log/{uid}/log.json', 'r', encoding='utf-8') as f: # 기록이 있다면 챗 로그 json 불러 옴
                chat_history = json.load(f)
                
        chat_history["chat"].append({  #기록에 유저의 질문 붙임
            "role": "user",
            "content": message
            })
        
        result = self.llm.create_chat_completion(
            max_tokens=512,
            frequency_penalty= 1,
            repeat_penalty= 1.1,
            temperature= 0.8,
            messages = list(chat_history["chat"])
            )
        
        #생성된 답변 추가
        chat_history["chat"].append({
            "role": "assistant",
            "content": result['choices'][0]['message']['content']
            })
        
        with open(f'./chat_log/{uid}/log.json', 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent="\t")
        
        print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
        return result['choices'][0]['message']['content']
        
        
        
        
    def llama_daily_chat(self, message, uid):
        if uid not in os.listdir('./daily_chatlog'): #처음 시도일 경우
            os.makedirs(f'./daily_chatlog/{uid}')   #uid로 경로 생성
            newchat ={
                "uid": uid,
                "chat":[
                    {
                        "role": "system",
                        "content": "너는 학생의 말에 공감하거나 안부를 묻는 상담하는 휼룡한 심리 상담사 이다. 오직 질문에 답변만 하라"
                    },
                    ]
                }
        
            with open(f'./daily_chatlog/{uid}/daily_log.json', 'w', encoding='utf-8') as f:
                json.dump(newchat, f, ensure_ascii=False, indent="\t")
        
             
        
        with open(f'./daily_chatlog/{uid}/daily_log.json', 'r', encoding='utf-8') as f: # 기록이 있다면 챗 로그 json 불러 옴
                chat_history = json.load(f)
        
        
        chat_history["chat"].append({
            "role": "user",
            "content": f"{message}"
            })
        
        print(chat_history["chat"],"\n",type(chat_history["chat"]))
        
        result = self.llm.create_chat_completion(
            max_tokens=512,
            frequency_penalty= 1,
            repeat_penalty= 1.1,
            temperature= 0.7,
            messages = chat_history["chat"]
            )
        
        chat_history["chat"].append({
            "role": "assistant",
            "content": result['choices'][0]['message']['content']
            })
        print(chat_history)
        with open(f'./daily_chatlog/{uid}/daily_log.json', 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent="\t")
            
        print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
        return result['choices'][0]['message']['content']
    
    
    
    def llama_analyze_emotion(self, uid):
        log = ""
        with open(f'./daily_chatlog/{uid}/daily_log.json', 'r', encoding='utf-8') as f:
            message = json.load(f)
        for i in message["chat"][1:]:
            log += f'{i["role"]} : {i["content"]} \n'
            
        result = self.llm.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": '''당신은 감정 분석을 하는 전문가입니다. 입력받은 대화내용을 분석하고 여기서 전반적인 user의 감정을 한단어로 분석하고 결과를 JSON으로 출력합니다. 출력 형식은 다음과 같습니다:

                    - "conclusion" : 대화내용의 전반적인 주제와 'user'의 감정을 요약하여 입력합니다.
                    - "emotion" : 대화내용에서 감정을 "분노", "슬픔", "기쁨", "중립", "불안", "걱정", "당황", "상처" 중 하나를 입력합니다. (예: "분노", "슬픔", "기쁨", "중립", "불안", "걱정", "당황", "상처").

                    ''',
                },
                {"role": "user", "content": f"{log}"},
            ],
            max_tokens=128,
            frequency_penalty= 1,
            repeat_penalty= 1.1,
            temperature= 0.8,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"conclusion": {"type": "string"},
                                   "emotion":{"type": "string"}
                                   },
                    "required": ["conclusion", "emotion"],
                }}
                    )
        print(f'답변: %s \n\n입력된 토큰:%d\n출력된 토큰:%d' %(result['choices'][0]['message']['content'], result['usage']['prompt_tokens'], result['usage']['completion_tokens']))
        return result['choices'][0]['message']['content']
            





# import fitz

# pdf_document = "어린왕자.pdf"
# doc = fitz.open(f'./{pdf_document}')

# # Initialize an empty string to store extracted text
# extracted_text = ""

# # Iterate through each page and extract text
# for page_num in range(doc.page_count):
#     page = doc[page_num]
#     extracted_text += page.get_text()

# # Close the PDF document
# doc.close()
# extracted_text=extracted_text.replace("\n","")


# chatbot = model()
# with open('.\lucky_dayTXT.txt', 'r', encoding='utf-8') as f:
#     extra = f.read()
# extra = "김첨지는 조선의 인력거꾼이며 나는 40살이고 돈을 벌기위해 비가오는 날에도 일을 하러 나갑니다. 그에게는 병든 아내가 있고 병원비를 벌기위해 열심히 일을 합니다. 그에게는 '치삼'이라는 친구가 있으며 가끔만나면 술을 마십니다."
# with open('./daily_chatlog/qwerty/daily_log.json', 'r', encoding='utf-8') as f:
#     message = json.load(f)
# log = ""
# for i in message["chat"][1:]:
#     log += f'{i["role"]} : {i["content"]} \n'
# print(log)
# message ="왜 같은 말만 반복하니?"
# res = chatbot.llama_daily_chat(message="안녕?", uid="zxc")
# with open('lucky_dayTXT.txt', 'r', encoding='utf-8') as f:
#     message = f.read()
# res = chatbot.llama_make_question_3(extra)
# print(res)
