import os
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. إعداد مفتاح Gemini API
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("الرجاء إعداد GEMINI_API_KEY في إعدادات المنصة (Secrets).")

model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="نظام البحث الذكي للمكتبة (RAG)", layout="wide")

# 2. القائمة الجانبية لرفع المستندات وفهرستها
st.sidebar.header("📁 إدارة المستندات")
uploaded_file = st.sidebar.file_uploader("لرفع PDF اختر ملف", type=["pdf"])

persist_directory = "vector_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if uploaded_file is not None:
    # حفظ الملف مؤقتًا على الخادم
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"تم رفع الملف: {uploaded_file.name}")

    if st.sidebar.button("فهرسة المستندات"):
        with st.spinner("جاري قراءة وفهرسة المستند..."):
            try:
                # قراءة وتجزئة النص
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # إنشاء قاعدة البيانات المتجهة وحفظها
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings, 
                    persist_directory=persist_directory
                )
                st.sidebar.success("تمت الفهرسة بنجاح!")
            except Exception as e:
                st.sidebar.error(f"حدث خطأ أثناء الفهرسة: {e}")

# 3. واجهة البحث الرئيسية
st.title("📚 (RAG + LLM) نظام البحث الذكي والتوليد الآلي")
st.write("اطرح سؤالاً، وسيقوم النظام بالبحث وتوليد إجابة نصية ذكية من ملفاتك.")

user_query = st.text_input("اكتب سؤالك هنا:")

if user_query:
    with st.spinner("جاري البحث وتوليد الإجابة..."):
        try:
            context = ""
            # التحقق من وجود قاعدة بيانات مفهرسة للبحث فيها
            if os.path.exists(persist_directory):
                vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(user_query)
                context = "\n".join([doc.page_content for doc in relevant_docs])

            # دمج السياق المستخرج مع سؤال المستخدم وإرساله لـ Gemini
            prompt = f"استناداً إلى المعلومات التالية:\n{context}\n\nأجب عن السؤال التالي: {user_query}"
            response = model.generate_content(prompt)
            
            st.subheader("💡 الإجابة الذكية:")
            st.write(response.text)
            
            if context:
                with st.expander("عرض المقاطع الأصلية المستخرجة من الملف"):
                    st.write(context)
                    
        except Exception as e:
            st.error(f"حدث خطأ أثناء الاتصال بالنموذج: {e}")