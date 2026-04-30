import streamlit as st
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
from typing import Optional

st.set_page_config(
    page_title="Legal QA System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("⚙️ Cấu hình")
st.sidebar.write("---")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Nhập API key của bạn từ OpenAI"
)

model = st.sidebar.selectbox(
    "Chọn model",
    ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    help="Chọn model GPT để sử dụng"
)

temperature = st.sidebar.slider(
    "Temperature (Độ sáng tạo)",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1,
    help="Cao hơn = sáng tạo hơn, Thấp hơn = chính xác hơn"
)

max_tokens = st.sidebar.slider(
    "Max Tokens (Độ dài trả lời)",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

st.sidebar.write("---")

st.title("⚖️ Hệ thống Hỏi-Đáp Pháp Luật")
st.write("Nhập câu hỏi về pháp luật và nhận câu trả lời từ AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

st.write("---")
st.subheader("📝 Nhập câu hỏi")

question = st.text_area(
    "Câu hỏi của bạn",
    placeholder="Ví dụ: Thủ tục xin phép xây dựng cần những tài liệu nào?",
    height=100,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns(3)

with col1:
    submit_button = st.button("🚀 Gửi câu hỏi", use_container_width=True, type="primary")

with col2:
    clear_button = st.button("🗑️ Xóa lịch sử", use_container_width=True)

with col3:
    st.write("")

if clear_button:
    st.session_state.messages = []
    st.session_state.last_response = ""
    st.rerun()

if submit_button:
    if not api_key:
        st.error("❌ Vui lòng nhập API key OpenAI trong thanh bên")
    elif not question.strip():
        st.error("❌ Vui lòng nhập câu hỏi")
    else:
        client = OpenAI(api_key=api_key)
        
        with st.spinner("⏳ Đang xử lý câu hỏi..."):
            try:
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                
                system_message = {
                    "role": "system",
                    "content": """Bạn là một trợ lý pháp lý chuyên nghiệp người Việt Nam.
                    
Nhiệm vụ của bạn:
- Trả lời câu hỏi về pháp luật Việt Nam một cách chính xác và chi tiết
- Giải thích các điều luật, quy định một cách dễ hiểu
- Cung cấp thông tin có cơ sở pháp lý
- Nếu không chắc chắn, hãy nói rõ điều đó
- Không cung cấp tư vấn pháp lý chuyên nghiệp, mà chỉ cung cấp thông tin giáo dục

Định dạng trả lời:
- Sử dụng Tiếng Việt
- Trình bày rõ ràng, có cấu trúc
- Sử dụng bullet points nếu cần thiết
- Kết thúc bằng disclaimer nếu cần"""
                }
                
                messages_to_send = [system_message] + st.session_state.messages
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages_to_send,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                assistant_message = response.choices[0].message.content
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                st.session_state.last_response = assistant_message
                
                st.rerun()
                
            except AuthenticationError:
                st.error("❌ API Key không hợp lệ. Vui lòng kiểm tra lại.")
            except RateLimitError:
                st.error("❌ Bạn đã vượt quá giới hạn API. Vui lòng thử lại sau.")
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")

if st.session_state.messages:
    st.write("---")
    st.subheader("💬 Lịch sử trò chuyện")
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                st.write(msg["content"])

st.write("---")
st.caption("💡 Lưu ý: Hệ thống này chỉ cung cấp thông tin giáo dục, không phải tư vấn pháp lý chuyên nghiệp.")
