import openai
import json
import os
from pathlib import Path
import requests
import threading
import customtkinter as ctk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import tkinter as tk
from queue import Queue, Empty
import time
import tiktoken
from tkinter import font
from get_embd import order_document_sections_by_query_similarity, get_embd

openai.api_key = ''

# Initialization
conversation_history = []
user_profile = {}
HISTORY_FILE = 'history.json'
USER_PROFILE_FILE = 'user_profile.txt'
UPDATE_PROFILE_EVERY = 5

def save_conversation_to_file(conversation_history):
    def remove_surrogates(input_str):
        return "".join(ch for ch in input_str if not ('\uD800' <= ch <= '\uDFFF'))
    with open(HISTORY_FILE, 'a+', encoding='utf-8') as file:
        file.seek(0)
        data = file.read(100)
        for message in conversation_history:
            if data:
                file.write("\n")
            cleaned_content = remove_surrogates(message['content'])
            message['content'] = cleaned_content
            json.dump(message, file, ensure_ascii=False)

def load_conversation_history():
    with open(HISTORY_FILE, 'r', encoding='utf-8') as file:
        history = []
        for line in file:
            try:
                history.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line}. Error: {e}")
        return history
    
def send_greeting_based_on_profile():
    prompt = f"根据以下的用户资料，请用中文提供一个友好的问候消息，并说表达我可以为你解答问题：{user_profile}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=300,
        stop=None,
        temperature=0.7
    )

    greeting_msg = response.choices[0].message['content'].strip()
    conversation_history.append({"role": "assistant", "content": greeting_msg})
    print(f"GPT：{greeting_msg}")
    return greeting_msg
    
def analyze_user_profile(conversation_history):
    if os.path.exists(USER_PROFILE_FILE):
        # Load the user profile from the file as plain text
        with open(USER_PROFILE_FILE, 'r', encoding='utf-8') as file:
            user_profile = file.read()
    else:
        combined_history = " ".join([msg["content"] for msg in conversation_history])
        messages = [
            {
                "role": "user",
                "content": f"Based on the following conversation history, please analyze and create a user profile detailing their interests, preferences, and other discernible characteristics in JSON format with keys 'interests', 'preferences', and 'characteristics': '{combined_history}' The final analysis needs to be general, merge things in common"
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        user_profile = (response.choices[0].message['content'])
        # Save the generated profile to the file
        with open(USER_PROFILE_FILE, 'w', encoding='utf-8') as file:
            json.dump(user_profile, file, ensure_ascii=False, indent=4)
        print(user_profile)
    return user_profile

def update_user_profile(conversation_history):
    user_profile_content = Path(USER_PROFILE_FILE).read_text(encoding='utf-8') if Path(USER_PROFILE_FILE).is_file() else ""
    combined_history = " ".join([msg["content"] for msg in conversation_history])
    messages = [
        {
            "role": "user",
            "content": 
            f"Given the existing user profile {user_profile_content} and the following new conversations this time , please update the user profile detailing their interests, preferences, and other discernible characteristics in JSON format with keys 'interests', 'preferences', and 'characteristics': '{combined_history}' \n."
            f"The user profile must be consise and accurate. If you find something are the same meaning or repeated, just merge them."
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    user_profile = response.choices[0].message['content'].strip()
    if user_profile:
        with open(USER_PROFILE_FILE, 'w', encoding='utf-8') as file:
            print('Successfully update the user profile!')
            file.write(user_profile)

def analyze_content(messages):
    content = " ".join([msg["content"] for msg in messages[-3:]])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Analyze the main topic(most important topic in current context, should be just one or two), detailed sentiment (such as joy, anger, surprise, sadness, anticipation, trust, disgust, fear, or other nuanced emotions), user intent of the following content, and suggest whether an image ('yes' or 'no') would enhance the conversation. Please provide the results in JSON format with keys 'topic', 'sentiment', 'intent', and 'image_suggestion'"},
            {"role": "user", "content": content}
        ]
    )

    return response.choices[0].message['content']

def parse_analysis_response(analysis):
    try:
        analysis = json.loads(analysis)
        topic = analysis.get("topic", "unknown topic")
        sentiment = analysis.get("sentiment", "neutral sentiment")
        intent = analysis.get("intent", "unknown intent")
        image_suggestion = analysis.get("image_suggestion", "no")
    except json.JSONDecodeError:
        topic, sentiment, intent, image_suggestion = "unknown topic", "neutral sentiment", "unknown intent", "no"
    print(topic, sentiment, intent, image_suggestion)
    return topic, sentiment, intent, image_suggestion

def generate_image_with_dalle():
    '''
    topic, sentiment, intent, image_suggestion = parse_analysis_response(analysis)
    _prompt = (
        f"Given the user profile {user_profile} and history conversation:{conversation_history}"
        f"Create an image that represents the topic '{topic}' and resonates with the sentiment '{sentiment}'. "
        f"The image should align with the intent '{intent}' and be suitable for enhancing the conversation."
    )
    '''
    content = " ".join([msg["content"] for msg in conversation_history[-2:]])
    _prompt = (f"The image that suits the current conversation most: {content}")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(_prompt)
    num_tokens = len(tokens)
    if num_tokens*2 > 1000:
        truncated_tokens = tokens[:420]
        _prompt = encoding.decode(truncated_tokens)
    response = openai.Image.create(
    prompt=_prompt,
    n=1,
    size="512x512"
    )
    image_url = response['data'][0]['url']
    if image_url:
        print(f"Generated image URL: {image_url}")
        return image_url
    else:
        print(f"Failed to generate image:")
        return None

def initiate_proactive_response(analysis, force_proactive=False):
    topic, sentiment, intent, image_suggestion = parse_analysis_response(analysis)

    if force_proactive:
        prompt = [{
            "role": "system",
            "content": (
                f"Given the user profile {user_profile}, topic '{topic}', sentiment '{sentiment}', intent '{intent}', and the following conversation history, initiate a conversation which would be helpful for users."
                f"The answer must be meaningful and new to the users. Ensure your response is entirely in Chinese. Here are the key points to consider: \n"
                f"- Provide a response in the json format: '{{\"decision\": \"yes\", \"message\": \"Your proactive message here.\"}}'.\n"
                f"- Do not repeat content from previous conversations. The message should initiate a new topic or content that would interest the user and be detailed and helpful.\n"
                f"- Make the your message detailed and helpful.\n"
                f"- If the sentiment is markedly positive or negative, ensure the message resonates with the user's emotions. For neutral or undetermined sentiments, make the message engaging and thought-provoking to invoke a clearer emotional response.\n"
                f"- Previous conversation: {conversation_history}"
            )
        }]
    else:
        prompt = [{
            "role": "system",
            "content": (
                f"Given the user profile {user_profile}, topic '{topic}', sentiment '{sentiment}',intent '{intent}', and the following conversation history, evaluate if initiating a conversation would be helpful for users. "
                f"Provide a response in the format: '{{\"decision\": \"yes/no\"}} If you answer is yes, provide a proactive message in the format: '{{\"message\": \"Your proactive message here.\"}}'.\n"
                f"If you decide to answer, the answer must be meaningful and new to the users. If you don't think it is necessary to answer, then say no. \n"
                f"Ensure your response is entirely in Chinese. Here are the key points to consider: \n"
                f"- The message should initiate a new topic or content that would interest the user and be detailed and helpful.\n"
                f"- Avoid repeating content from previous conversations.\n"
                f"- If the sentiment is markedly positive or negative, ensure the message resonates with the user's emotions. For neutral or undetermined sentiments, make the message engaging and thought-provoking to invoke a clearer emotional response.\n"
                f"- Previous conversation: {conversation_history}"
            )
        }]

    decision_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=500,
        stop=None,
        temperature=0.7
    )

    model_decision_response = decision_response.choices[0].message['content'].strip()
    print(decision_response)
    def repair_json_output(output_str):
        try:
            json.loads(output_str)
            return output_str
        except json.JSONDecodeError:
            repaired_str = output_str.rstrip().rstrip(",") + '"}'
            return repaired_str
    repaired_output = repair_json_output(model_decision_response)
    try:
        decision_data = json.loads(repaired_output)
        if decision_data["decision"] == "yes":
            conversation_history.append({"role": "assistant", "content": decision_data["message"]})
            return decision_data["message"], image_suggestion
        else:
            return None, image_suggestion
    except json.JSONDecodeError:
        return None, image_suggestion
    
def proactive_conversation(analysis, user_input="", most_relevant_document_sections=""):
    conversation_history.append({"role": "user", "content": user_input})
    topic, sentiment, intent, image_suggestion = parse_analysis_response(analysis)

    prompt = (
        f"Given the user profile {user_profile}, conversation history{conversation_history}, relevant sections found in the database{most_relevant_document_sections}, and considering the topic '{topic}', "
        f"sentiment '{sentiment}', and intent '{intent}', craft a response. "
        f"If the sentiment is markedly positive or negative, provide feedback that resonates with the user's emotions. "
        f"For neutral or undetermined sentiments, aim to make the response more engaging and thought-provoking to invoke a clearer emotional response. "
        f"User's input: '{user_input}'"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500,
        stop=None,
        temperature=0.7
    )

    model_response = response.choices[0].message['content'].strip()
    conversation_history.append({"role": "assistant", "content": model_response})

    return model_response, image_suggestion

def get_relevant_documents(user_input, df, document_embeddings):
    most_relevant_document_sections = order_document_sections_by_query_similarity(user_input, document_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    count=0
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        chosen_sections_len += len(document_section.content)
        if count>2:
            break
        chosen_sections.append( ": " + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        count+=1
    return "".join(chosen_sections)

# Define global variables for the UI
conversation_textbox = None
user_input_entry = None
root = None
input_queue = Queue()  # Queue for handling input

def initiate_proactive_conversation():
    input_queue.put("Q")
    
def send_user_input(event=None):
    user_input = user_input_entry.get()
    user_input_entry.delete(0, tk.END)  # Clear the input field after sending the message
    update_conversation_ui(user_input=user_input)
    input_queue.put(user_input)

def update_conversation_ui(user_input=None, response=None, image_url=None):
    if user_input:
        conversation_textbox.insert(tk.END, f"You: {user_input}\n")

    if response:
        conversation_textbox.insert(tk.END, f"GPT: {response}\n")
    customFont = font.Font(family="Microsoft YaHei", size=14)
    conversation_textbox.configure(font=customFont)
    if image_url:
        # Load and display the image from the URL
        image = Image.open(requests.get(image_url, stream=True).raw)
        photo = ImageTk.PhotoImage(image)
        if not hasattr(conversation_textbox, 'photo_list'):
            conversation_textbox.photo_list = []
        conversation_textbox.photo_list.append(photo)
        conversation_textbox.image_create(tk.END, image=photo)
        conversation_textbox.insert(tk.END, '\n')

    conversation_textbox.see(tk.END)  # Scroll to the end

def center_window(w, h):
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))

def ui_thread():
    # Create the Tkinter window
    global root, conversation_textbox, user_input_entry
    root = ctk.CTk()
    root.title("GPT Conversation")
    center_window(600, 500)
    conversation_textbox = ScrolledText(root, width=100, height=30)
    conversation_textbox.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    user_input_entry = tk.Entry(root, width=100)
    user_input_entry.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
    customFont = font.Font(family="Microsoft YaHei", size=14)
    user_input_entry.configure(font=customFont)
    proactive_button = tk.Button(root, text="Initiate Proactive Conversation", command=initiate_proactive_conversation)
    proactive_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")
    send_button = tk.Button(root, text="Send", command=send_user_input)
    send_button.grid(row=2, column=1, padx=10, pady=5, sticky="e")
    root.mainloop()

def main():
    global user_profile
    threading.Thread(target=ui_thread, daemon=True).start()
    print("开始与GPT进行深入的对话...")
    _conversation_history = load_conversation_history()
    user_profile = analyze_user_profile(_conversation_history)
    greeting_msg = send_greeting_based_on_profile()
    root.after(0, lambda: update_conversation_ui(response=greeting_msg))
    user_input = ""
    message_count = 0
    while True:
        try:
            user_input = input_queue.get_nowait()
        except Empty:
            time.sleep(0.1)
            continue
        print(user_input)
        # 主动对话
        if user_input.strip().upper() == "Q":
            raw_analysis_response = analyze_content(conversation_history)
            proactive_msg = initiate_proactive_response(raw_analysis_response, force_proactive=True)
            root.after(0, lambda: update_conversation_ui(response=proactive_msg))
            print(f"GPT：{proactive_msg}")
            continue
        # 结束对话
        if user_input.strip().upper() == "E":
            save_conversation_to_file(conversation_history)
            update_user_profile(conversation_history)
            print("对话历史已保存。再见！")
            break
        df, document_embeddings=get_embd()
        most_relevant_document_sections = get_relevant_documents(user_input, df, document_embeddings)
        print(most_relevant_document_sections)
        raw_analysis_response = analyze_content(conversation_history)
        response, image_suggestion = proactive_conversation(raw_analysis_response, user_input)
        root.after(0, lambda: update_conversation_ui(response=response))
        print(f"GPT：{response}")
        # 判断是否生成图片
        if image_suggestion == "yes":
            image_url = generate_image_with_dalle(raw_analysis_response)
            root.after(0, lambda: update_conversation_ui(image_url=image_url))
        raw_analysis_response = analyze_content(conversation_history)
        proactive_msg, image_suggestion = initiate_proactive_response(raw_analysis_response)
        if proactive_msg:
            print(f"GPT：{proactive_msg}")
            root.after(0, lambda: update_conversation_ui(response=proactive_msg))
        if image_suggestion == "yes":
            image_url = generate_image_with_dalle(raw_analysis_response)
            root.after(0, lambda: update_conversation_ui(image_url=image_url))
        # 更新profile
        message_count += 1
        if message_count % UPDATE_PROFILE_EVERY == 0:
            update_user_profile(conversation_history)

if __name__ == "__main__":
    main()