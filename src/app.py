import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import HashingVectorizer
import streamlit as st

VECTOR_DB_DIR = "./vector_db"
DOCS_DIR = "./data"


class FixedHashingEmbeddings(Embeddings):

  def __init__(self, n_features=384):
    self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    matrix = self.vectorizer.transform(texts)
    return matrix.toarray().tolist()

  def embed_query(self, text: str) -> list[float]:
    matrix = self.vectorizer.transform([text])
    return matrix.toarray()[0].tolist()


def clean_arabic_text(text: str) -> str:
  if not text:
    return ""
  lines = [line.strip() for line in text.split("\n") if line.strip()]
  return " ".join(lines)


def query_ollama_llm(context: str, query: str) -> str:
  prompt = f"""بناءً على النص المستخرج التالي، أجب عن السؤال باختصار ودقة باللغة العربية:

النص المستخرج:
{context}

السؤال: {query}
الإجابة الذكية:"""

  try:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
        timeout=120,
    )
    if response.status_code == 200:
      return response.json().get("response", "لم يتم استرجاع إجابة.")
    else:
      return f"⚠️ خطأ في الاستجابة (رمز الحالة: {response.status_code})"
  except Exception as e:
    return f"⚠️ تجاوزت مهلة الانتظار أو تعذر الاتصال بـ Ollama. التفاصيل: {e}"


st.set_page_config(
    page_title="نظام البحث الذكي في المكتبة", page_icon="📚", layout="wide"
)
st.title("📚 نظام البحث الذكي والتوليد الآلي (RAG + LLM)")

st.sidebar.header("📁 إدارة المستندات")
uploaded_file = st.sidebar.file_uploader("اختر ملف PDF لرفعه", type=["pdf"])

if uploaded_file is not None:
  if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

  file_path = os.path.join(DOCS_DIR, uploaded_file.name)
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.sidebar.success(f"تم رفع الملف: {uploaded_file.name}")

  if st.sidebar.button("🔄 فهرسة المستندات"):
    with st.spinner("جاري قراءة وفهرسة الملفات..."):
      all_documents = []
      pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]

      for file in pdf_files:
        f_path = os.path.join(DOCS_DIR, file)
        loader = PyPDFLoader(f_path)
        all_documents.extend(loader.load())

      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=400, chunk_overlap=40
      )
      texts = text_splitter.split_documents(all_documents)

      embeddings = FixedHashingEmbeddings(n_features=384)
      vectorstore = Chroma.from_documents(
          texts, embeddings, persist_directory=VECTOR_DB_DIR
      )
      st.sidebar.success("✅ تمت فهرسة جميع المستندات بنجاح!")

st.write("اطرح سؤالاً، وسيقوم النظام بالبحث وتوليد إجابة نصية ذكية من ملفاتك:")

if os.path.exists(VECTOR_DB_DIR):
  embeddings = FixedHashingEmbeddings(n_features=384)
  vectorstore = Chroma(
      persist_directory=VECTOR_DB_DIR, embedding_function=embeddings
  )

  query = st.text_input("اكتب سؤالك هنا:")

  if query:
    with st.spinner(
        "جاري البحث واستخلاص الإجابة السريعة عبر Ollama (قد يستغرق بعض الوقت"
        " لأول مرة)..."
    ):
      try:
        # تقليل عدد المقاطع المسترجعة إلى 1 لسرعة الاستجابة وتجنب الـ Timeout
        docs = vectorstore.similarity_search(query, k=1)

        if docs:
          cleaned_docs = [clean_arabic_text(doc.page_content) for doc in docs]
          context_text = "\n\n".join(cleaned_docs)

          ai_answer = query_ollama_llm(context_text, query)

          st.subheader("💡 الإجابة الذكية:")
          st.success(ai_answer)

          with st.expander("📄 عرض المقطع الأصلي المنظف"):
            for i, text in enumerate(cleaned_docs):
              st.info(f"**المقطع المطابق:**\n\n{text}")
        else:
          st.warning("لم يتم العثور على نتائج مطابقة للسؤال.")
      except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
else:
  st.warning("⚠️ يرجى رفع ملف PDF وفهرسته من القائمة الجانبية للبدء.")