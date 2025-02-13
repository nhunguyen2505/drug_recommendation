from flask import Flask, request, jsonify, render_template,make_response,redirect, url_for, session
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bcrypt
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import json
import os
import pickle
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import uuid

# Khởi tạo Flask app
app = Flask(__name__)

app.secret_key = 'AIzaSyCvlZ63Nkt5NpjdmxYPAsG8Qskex6usCFw'
app.config['MONGO_URI'] = 'mongodb+srv://hoangsontruonghcm:dsu3mMO944XuNut7@cluster0.2mwhu.mongodb.net/drug_recom'
mongo = PyMongo(app)




CORS(app)

genai.configure(api_key="AIzaSyCvlZ63Nkt5NpjdmxYPAsG8Qskex6usCFw")

# Tải FAISS index
faiss_index = faiss.read_index("source/faiss_index_vn.bin")

# Tải embeddings
with open("source/sentence_embeddings_vn.pkl", "rb") as f:
    disease_embeddings = pickle.load(f)

# Khởi tạo mô hình Sentence Transformer
model_em = SentenceTransformer('hiieu/halong_embedding')
# Đọc dữ liệu bệnh
merged_df = pd.read_csv('source/merged_df_vn.csv')


def get_disease_and_generate_prompt(symptoms_input, faiss_index, model_em, merged_df, top_k=5):
    # 1. Mã hóa triệu chứng đầu vào
    input_embedding = model_em.encode([symptoms_input], convert_to_tensor=True)

    # 2. Tìm kiếm top_k trong FAISS Index
    distances, indices = faiss_index.search(np.array(input_embedding.cpu().numpy()), k=top_k)

    # Lấy danh sách các bệnh và điểm tương ứng
    top_diseases = [(merged_df.iloc[idx], score) for idx, score in zip(indices[0], distances[0])]

    # 3. Mã hóa thông tin của top_k bệnh để đánh giá lại
    candidate_embeddings = model_em.encode(
        [disease['Information'] for disease, _ in top_diseases],
        convert_to_tensor=True
    )

    # 4. Tính độ tương đồng cosine giữa triệu chứng đầu vào và các ứng viên
    scores = util.cos_sim(input_embedding, candidate_embeddings).squeeze()

    # 5. Sắp xếp lại danh sách ứng viên dựa trên điểm similarity
    ranked_indices = scores.argsort(descending=True)
    best_match = top_diseases[ranked_indices[0].item()][0]  # Ứng viên tốt nhất sau re-ranking

     # 6. Chuyển thông tin bệnh tốt nhất thành danh sách
    result_list = [
        f"Patient Symptoms: {symptoms_input}",
        f"similarity: {scores}",
        f"disease: {best_match['Disease']}",
        f"symptoms: {best_match['Symptoms']}",
        f"medications: {best_match['Medication']}",
        f"diets: {best_match['Diet']}",
        f"workouts: {best_match['workout']}",
        f"precautions: {best_match.get('Precaution_1', '')}, {best_match.get('Precaution_2', '')}, {best_match.get('Precaution_3', '')}, {best_match.get('Precaution_4', '')}",
    ]

    return result_list



def parse_contexts(raw_contexts):
    """Convert raw context strings into structured dictionaries."""
    structured_contexts = []
    temp_context = {}
    for context in raw_contexts:
        # Check for each expected piece of information and assign it to the dictionary
        if context.startswith("Patient Symptoms:"):
            temp_context["patient_symptoms"] = context.replace("Patient Symptoms:", "").strip()
        elif context.startswith("disease"):
            temp_context["disease"] = context.replace("disease", "").strip()
        elif context.startswith("symptoms:"):
            temp_context["symptoms"] = context.replace("symptoms:", "").strip().split(", ")
        elif context.startswith("medications:"):
            temp_context["medications"] = context.replace("medications:", "").strip().split(", ")
        elif context.startswith("diets:"):
            temp_context["diets"] = context.replace("diets:", "").strip().split(", ")
        elif context.startswith("workouts:"):
            temp_context["workouts"] = context.replace("workouts:", "").strip().split(", ")
        elif context.startswith("precautions:"):
            temp_context["precautions"] = context.replace("precautions:", "").strip().split(", ")

        # When all necessary fields are collected, add to the list and reset temp_context
        if len(temp_context) == 7:
            structured_contexts.append(temp_context)
            temp_context = {}

    return structured_contexts

prompt_template = (
    "### Hệ thống:"
    "Bạn đang nhận được một yêu cầu tư vấn y tế. Dưới đây là thông tin cần thiết để bạn đưa ra câu trả lời chính xác, dễ hiểu và hữu ích cho người dùng."
    "Hãy tập trung vào việc cung cấp tên thuốc phù hợp và hướng dẫn rõ ràng để giúp người dùng áp dụng dễ dàng."
    "### Hướng dẫn:"
    "{instruction}\n\n"

    "### Thông tin y tế:\n" 
    "{input}\n\n"

    "### Câu trả lời:\n" 
    "{output}" 
    )

def get_prompt(question, raw_contexts):
    if not raw_contexts:
        raise ValueError("Danh sách thông tin y tế không được để trống.")

    # Xử lý dữ liệu đầu vào thành dạng dễ đọc
    contexts = parse_contexts(raw_contexts)
    
    context = "".join([
        f"\n<b>📌 Trường hợp {i+1}:</b>\n"
        f"- <b>Bệnh:</b> {x.get('disease', 'Chưa xác định')}\n"
        f"- <b>Triệu chứng:</b> {', '.join(map(str, x.get('symptoms', [])))}\n"
        f"- <b>Thuốc đề xuất:</b> <i>{', '.join(map(str, x.get('medications', [])))}</i>\n"
        f"- <b>Chế độ ăn uống:</b> {', '.join(map(str, x.get('diets', [])))}\n"
        f"- <b>Bài tập hỗ trợ:</b> {', '.join(map(str, x.get('workouts', [])))}\n"
        f"- <b>Lưu ý quan trọng:</b> {', '.join(map(str, x.get('precautions', [])))}\n"
        for i, x in enumerate(contexts)
    ])

    instruction = (
        "💊 Bạn là một dược sĩ có kinh nghiệm lâu năm. Hãy cung cấp câu trả lời <b>đầy đủ, chính xác</b>, "
        "dễ hiểu và tập trung vào <b>tư vấn thuốc</b> cho người dùng.\n"
        "🔹 Trình bày câu trả lời theo danh sách số thứ tự (1️⃣, 2️⃣, 3️⃣...) để dễ đọc.\n"
        "🔹 Định dạng rõ ràng: <b>in đậm</b> những điểm quan trọng, <i>in nghiêng</i> tên thuốc, <b>in đậm</b> các lưu ý quan trọng, "
        "sử dụng dấu gạch đầu dòng (-) để liệt kê thông tin." 
    )

    input_text = (
        "🩺 Dựa trên thông tin y tế sau đây, hãy trả lời câu hỏi của người dùng:\n"
        f"{context}\n"
        "❓ <b>Câu hỏi:</b> " + question + "\n"
        "📌 <b>Yêu cầu:</b> Hãy trả lời theo bố cục số thứ tự, dễ đọc, ngắn gọn nhưng đầy đủ, giúp người dùng dễ áp dụng.\n"
        "📋 <b>Định dạng:</b>\n"
        "- <b>In đậm</b> cho thông tin quan trọng\n"
        "- <i>In nghiêng</i> cho tên thuốc\n"
        "- <b>In đậm</b> cho các lưu ý đặc biệt\n"
        "- Dấu gạch đầu dòng (-) để trình bày rõ ràng khi liệt kê"
    )

    prompt = prompt_template.format(
        instruction=instruction,
        input=input_text,
        output=''  # AI sẽ tự điền câu trả lời
    )

    return prompt

@app.route("/send_message", methods=["POST"])
def send_message():
    try:
        data = request.json
        print("📩 Received Data:", data)  # Debug log

        # Kiểm tra request JSON hợp lệ
        if not data or "message" not in data:
            return jsonify({"error": "Dữ liệu không hợp lệ!"}), 400

        message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id", "")

        if not message:
            return jsonify({"error": "Câu hỏi không hợp lệ!"}), 400

        # Nếu không có conversation_id, tạo cuộc trò chuyện mới
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            session["conversation_id"] = conversation_id
            mongo.db.chat_history.insert_one({
                "conversation_id": conversation_id,
                "name": "Chưa có tên",
                "messages": [],
                "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            })

        # Gọi hàm lấy dữ liệu context
        context_data = get_disease_and_generate_prompt(message, faiss_index, model_em, merged_df, top_k=5)
        prompt = get_prompt(message, context_data)

        # Cấu hình model Gemini
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.6,
            "top_k": 40,
            "max_output_tokens": 3000,
            "response_mime_type": "text/plain",
        }

        # Gửi prompt đến model Gemini
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        # Lấy nội dung phản hồi từ bot
        try:
            text_response = response.candidates[0].content.parts[0].text
        except Exception as e:
            print(f"⚠️ Error getting response: {e}")
            return jsonify({"error": "Lỗi khi lấy phản hồi từ mô hình!"}), 500

        # Cấu trúc tin nhắn
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        user_message = {"text": message, "timestamp": timestamp, "sender": "user"}
        bot_message = {"text": text_response, "timestamp": timestamp, "sender": "bot"}

        # Cập nhật cuộc trò chuyện
        mongo.db.chat_history.update_one(
            {"conversation_id": conversation_id},
            {"$push": {"messages": {"$each": [user_message, bot_message]}}},
            upsert=True
        )

        # Trả về phản hồi JSON
        return jsonify({
            "conversation_id": conversation_id,
            "status": "sent",
            "bot_reply": text_response,
            "timestamp": timestamp
        })

    except Exception as e:
        print(f"🚨 Error in /query: {str(e)}")
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


@app.route("/start_conversation", methods=["POST"])
def start_conversation():
    try:
        data = request.json
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Tin nhắn không hợp lệ!"}), 400

        conversation_id = str(uuid.uuid4())  # Tạo conversation_id mới
        session["conversation_id"] = conversation_id

        mongo.db.chat_history.insert_one({
            "conversation_id": conversation_id,
            "name": message,
            "messages": [],
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        })

        return jsonify({"conversation_id": conversation_id})
    
    except Exception as e:
        print(f"🚨 Error in /start_conversation: {str(e)}")
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


@app.route("/all_history", methods=["GET"])
def get_all_history():
    """Lấy danh sách tất cả cuộc trò chuyện"""
    chats = list(mongo.db.chat_history.find({}, {"_id": 0, "messages": 0}))
    return jsonify(chats)


@app.route("/conversation/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    """Lấy nội dung cuộc trò chuyện theo ID"""
    chat = mongo.db.chat_history.find_one({"conversation_id": conversation_id}, {"_id": 0})
    if not chat:
        return jsonify({"error": "Cuộc trò chuyện không tồn tại"}), 404
    return jsonify(chat)

@app.route('/get_conversations', methods=['GET'])
def get_conversations():
    # Lấy tất cả cuộc trò chuyện từ MongoDB
    conversations = list(mongo.db.chat_history.find({}, {"_id": 0}))

    return jsonify([{
        "conversation_id": conv.get("conversation_id", "N/A"),
        "name": conv.get("name", "Chưa có tên"),
        "bot_messages": [msg["text"] for msg in conv.get("messages", []) if msg.get("sender") == "bot"],  # Chỉ lấy text của bot
        "created_at": conv.get("created_at", "Không rõ ngày")
    } for conv in conversations])

@app.route('/get_messages', methods=['GET'])
def get_messages():
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Thiếu conversation_id"}), 400

    # Truy vấn cuộc trò chuyện theo conversation_id
    conversation = mongo.db.chat_history.find_one(
        {"conversation_id": conversation_id}, 
        {"_id": 0, "messages": 1,"name":1, "timestamp":1}
    )

    if conversation:
        return jsonify(conversation)  # Trả về danh sách tin nhắn
    else:
        return jsonify({"error": "Không tìm thấy cuộc trò chuyện"}), 404



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        hash_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert user into the 'user' collection
        mongo.db.user.insert_one({'fullname': fullname,'username': username, 'password': hash_password})
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Find user in the 'user' collection
        user = mongo.db.user.find_one({'username': username})

        # Kiểm tra nếu người dùng tồn tại và mật khẩu đúng
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            # Lưu fullname của người dùng vào session
            session['user'] = user.get('fullname', 'Unknown User')  # Đảm bảo lấy giá trị 'fullname' nếu có
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid username or password'
    return render_template('login.html')
    
@app.route("/dashboard")
def dashboard():
    if 'user' in session:
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        username = session["user"]   
        return render_template('dashboard.html',username=username,current_time=current_time)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))



if __name__ == "__main__":
    app.run(debug=True)
