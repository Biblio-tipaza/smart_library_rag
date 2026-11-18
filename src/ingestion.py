import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import HashingVectorizer

VECTOR_DB_DIR = "./vector_db"
DOCS_DIR = "./data"


class FixedHashingEmbeddings(Embeddings):
  """كلاس متجهات يعتمد على حجم ثابت (Fixed Dimension) لتجنب أخطاء الأبعاد تماماً"""

  def __init__(self, n_features=384):
    # نحدد حجماً ثابتاً للمتجه (مثلاً 384) ليتطابق دائماً بغض النظر عن طول النص
    self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    matrix = self.vectorizer.transform(texts)
    return matrix.toarray().tolist()

  def embed_query(self, text: str) -> list[float]:
    matrix = self.vectorizer.transform([text])
    return matrix.toarray()[0].tolist()


def build_vector_database():
  all_documents = []

  if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    print("مجلد data غير موجود، تم إنشاؤه.")
    return

  pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
  if not pdf_files:
    print("⚠️ تنبيه: لم يتم العثور على أي ملف PDF داخل مجلد data!")
    return

  for file in pdf_files:
    file_path = os.path.join(DOCS_DIR, file)
    print(f"جاري قراءة الملف: {file}...")
    loader = PyPDFLoader(file_path)
    all_documents.extend(loader.load())

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  )
  texts = text_splitter.split_documents(all_documents)

  print(f"تم تقسيم النص إلى {len(texts)} مقطع. جاري الفهرسة الثابتة...")

  embeddings = FixedHashingEmbeddings(n_features=384)

  # حذف المجلد القديم إن وجد برمجياً لتجنب التعارض
  vectorstore = Chroma.from_documents(
      texts, embeddings, persist_directory=VECTOR_DB_DIR
  )
  print("✅ تمت الفهرسة بنجاح تام بحجم ثابت للمتجهات!")


if __name__ == "__main__":
  build_vector_database()