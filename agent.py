import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from typing import TypedDict, List
from pydantic import BaseModel, Field

# 1. Load Environment Variables
load_dotenv()

# 2. Inisialisasi Model
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("YOUR_API_KEY"),
    base_url=os.getenv("YOUR_BASE_URL"),
    temperature=0
) 

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("YOUR_API_KEY"),
    base_url=os.getenv("YOUR_BASE_URL"),
)

# 3. Helper Function untuk Extract Teks PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Menggabungkan seluruh halaman menjadi satu string teks utuh
        full_text = "\n\n".join([page.page_content for page in pages])
        print(f"Berhasil memuat PDF ({len(pages)} halaman).")
        return full_text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise

# --- 1. Schema Output Pydantic ---
class CandidateProfile(BaseModel):
    name: str = Field(description="Nama lengkap kandidat")
    skills: List[str] = Field(description="Daftar keahlian/skills utama")
    total_experience_years: float = Field(description="Estimasi total pengalaman kerja dalam tahun")
    education: str = Field(description="Pendidikan terakhir dan jurusan")

class EvaluationResult(BaseModel):
    match_score: int = Field(description="Skor kesesuaian dari 0 - 100")
    strengths: List[str] = Field(description="Poin kelebihan kandidat sesuai JD")
    weaknesses: List[str] = Field(description="Poin kekurangan / skill gap kandidat")
    recommendation: str = Field(description="Rekomendasi: PROCEED, HUMAN_REVIEW, atau REJECT")

# --- 2. Definition LangGraph State ---
class HRState(TypedDict):
    cv_text: str
    job_description: str
    candidate_profile: dict
    evaluation: dict

# --- Node 1: Ekstraksi Informasi CV ---
def extract_cv_info(state: HRState):
    print("\n[Node 1] Memproses ekstraksi data CV...")
    structured_llm = llm.with_structured_output(CandidateProfile)
    
    prompt = f"""
    Ekstrak informasi penting dari teks CV berikut ke dalam format terstruktur:
    
    TEKS CV:
    {state['cv_text']}
    """
    
    res = structured_llm.invoke(prompt)
    return {"candidate_profile": res.model_dump()}

# --- Node 2: Evaluasi & Scoring ---
def evaluate_candidate(state: HRState):
    print("[Node 2] Menganalisis kesesuaian dengan Job Description...")
    structured_llm = llm.with_structured_output(EvaluationResult)
    
    prompt = f"""
    Anda adalah HR Specialist senior. Evaluasi kesesuaian kandidat berikut dengan Job Description (JD).
    
    PROFIL KANDIDAT:
    {state['candidate_profile']}
    
    JOB DESCRIPTION:
    {state['job_description']}
    """
    
    res = structured_llm.invoke(prompt)
    return {"evaluation": res.model_dump()}

graph = StateGraph(HRState)

graph.add_node("extract_info", extract_cv_info)
graph.add_node("evaluate", evaluate_candidate)

# Hubungkan Nodes
graph.add_edge(START, "extract_info")
graph.add_edge("extract_info", "evaluate")
graph.add_edge("evaluate", END)

# Compile Graph
hr_app = graph.compile()

# --- 4. Cara Menjalankan (Testing) ---
if __name__ == "__main__":
    # 1. Ambil teks dari PDF (menggunakan fungsi helper sebelumnya)
    raw_cv = extract_text_from_pdf("cv.pdf")
    
    # 2. Tentukan Job Description
    sample_jd = """
    Dibutuhkan: Python Backend Developer
    Kualifikasi:
    - Pengalaman minimal 2 tahun menggunakan Python (FastAPI/Django).
    - Terbiasa dengan PostgreSQL dan Redis.
    - Memahami konsep REST API & microservices.
    - Pendidikan S1 Teknik Informatika atau setara.
    """
    
    # 3. Jalankan LangGraph
    initial_input = {
        "cv_text": raw_cv,
        "job_description": sample_jd
    }
    
    result = hr_app.invoke(initial_input)
    
    # 4. Tampilkan Hasil
    print("\n" + "="*40)
    print("PROFIL KANDIDAT:")
    print(result["candidate_profile"])
    print("\nHASIL EVALUASI HR:")
    print(result["evaluation"])
    print("="*40)