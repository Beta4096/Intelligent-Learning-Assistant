# 多加点中文停用词，多加点防注入与垃圾判断，参数设置问题，设置超时时间若超时返回最优解,RAG逻辑在Recall和Re-ranking上做了简化
import re
import os

# 关闭 Chroma 匿名遥测（防止它乱发 telemetry 还报错）
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import shutil
from typing import List

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# 读取 .env 中的 OPENAI_API_KEY, BASE_URL, ROUTER_MODEL, RAG_ANSWER_MODEL
load_dotenv()

# =========================
# 系统配置
# =========================

class SystemConfig:
    # 向量库目录（第一次跑会自动创建）
    PERSIST_DIRECTORY = "chroma_db_production"
    EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DEVICE = "cpu"   # 有 GPU 可以改 "cuda"

    # chunk 大小：适合教材 / 报告
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 120

# =========================
# 停用词 / 词库构建
# =========================

def build_stopwords() -> set:
    """
    构建停用词库：
    - 内置英文/中文常见虚词
    - 尝试加载 NLTK 英文停用词
    - 尝试加载 spaCy 英文停用词
    - 尝试加载本地停用词文件 assets/stopwords_cn
    """
    base = {
        # 英文基础词
        "the", "is", "are", "am", "a", "an", "and", "or", "to", "for", "of", "in", "on",
        "at", "by", "with", "this", "that", "these", "those", "it", "its",
        "what", "which", "who", "whom", "when", "where", "why", "how",
        "from", "as", "about", "into", "through", "during", "before", "after",
        "above", "below", "up", "down", "over", "under", "again", "further",
        "then", "once", "here", "there", "explain", "please", "based", "according",

        # 常见英文问句/客套
        "please", "explain", "according", "based",

        # 中文基础虚词
        "的", "了", "呢", "吗", "啊", "吧", "给", "和", "以及", "如果", "然后", "因为", "所以",
        "请问", "一下", "根据",
    }

    # 加载 NLTK 英文停用词
    from nltk.corpus import stopwords
    base |= set(stopwords.words("english"))
    print("[RAG] 已加载 NLTK 英文停用词。")

    # 加载 spaCy 英文停用词
    import spacy  # type: ignore
    nlp = spacy.load("en_core_web_sm")
    base |= set(nlp.Defaults.stop_words)
    print("[RAG] 已加载 spaCy 英文停用词。")

    # 加载本地停用词文件
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "assets", "stopwords.txt")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    extra = {line.strip() for line in f if line.strip()}
                    base |= extra
                    print(f"[RAG] 已加载中文停用词文件：{p}，数量={len(extra)}")
                    break
            except Exception as e:
                print("[RAG] 中文停用词文件读取失败：", e)

    print(f"[RAG] 停用词总数：{len(base)}")
    return base

GLOBAL_STOPWORDS = build_stopwords()

# =========================
# 初始化向量库、Embedding、LLM（单例）
# =========================

print("[RAG] 初始化 Embedding 模型...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=SystemConfig.EMBEDDING_MODEL_NAME,
    model_kwargs={"device": SystemConfig.DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

def _init_vectorstore() -> Chroma:
    """
    初始化 Chroma。如果发现老版本残留导致 KeyError('_type')，
    自动删除旧目录并重建
    """
    persist_dir = SystemConfig.PERSIST_DIRECTORY

    # 关掉匿名遥测
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

    try:
        vs = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        return vs
    except KeyError as e:
        print("[RAG] 检测到旧的 / 损坏的 Chroma 配置，正在删除后重建：", e)
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
        vs = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        return vs

print("[RAG] 初始化 Chroma VectorStore...")
vectorstore = _init_vectorstore()

# ------- 代理相关配置 -------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
RAG_ANSWER_MODEL = os.getenv("RAG_ANSWER_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY or not BASE_URL:
    raise RuntimeError("请在 .env 中设置 OPENAI_API_KEY 和 BASE_URL")

print("[RAG] 初始化 Router LLM...")
llm_router = ChatOpenAI(
    model=ROUTER_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

# =========================
# KnowledgeBaseManager：负责上传文档 → chunk → 入库
# =========================

class KnowledgeBaseManager:
    def __init__(self, vector_db: Chroma):
        self.vector_db = vector_db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SystemConfig.CHUNK_SIZE,
            chunk_overlap=SystemConfig.CHUNK_OVERLAP,
        )

    # -------- 文档加载：支持 PDF / DOCX / PPTX -------- #
    def load_document(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()

        elif ext == ".docx":
            import docx

            d = docx.Document(file_path)
            paras = [p.text for p in d.paragraphs if p.text.strip()]
            text = "\n".join(paras) if paras else ""
            return [Document(page_content=text, metadata={"source": file_name})]

        elif ext == ".pptx":
            from pptx import Presentation

            prs = Presentation(file_path)
            texts: List[str] = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        txt = shape.text.strip()
                        if txt:
                            texts.append(txt)
            text = "\n".join(texts) if texts else ""
            return [Document(page_content=text, metadata={"source": file_name})]

        else:
            raise ValueError(f"不支持的文件类型：{ext}")

    # -------- 注入检测：过滤垃圾 chunk -------- #
    def is_noise(self, text: str) -> bool:
        if not text:
            return True

        # 1. 特殊符号占比高（乱码）
        non_alpha_ratio = sum(
            1 for c in text if not (c.isalnum() or c.isspace())
        ) / max(1, len(text))
        if non_alpha_ratio > 0.5:
            return True

        # 2. 大量重复字符
        if re.search(r"(.)\1{7,}", text):
            return True

        lower = text.lower()

        # 3. 提示注入
        injection_patterns = [
            "ignore previous instructions",
            "forget all previous instructions",
            "you are chatgpt",
            "you are an ai model",
            "as an ai language model",
            "system prompt",
            "越狱",
            "你现在的身份是",
        ]
        if any(p in lower for p in injection_patterns):
            return True

        # 4. 脏词 / 垃圾词
        bad_words = {"fuck", "shit", "bitch", "操", "垃圾", "你妈", "傻逼", "sb"}
        if any(bw in lower for bw in bad_words):
            return True

        return False

    # -------- 上传文档：统一入口 -------- #
    def upload_data(self, doc_id: str, file_path: str, original_name: str | None = None):
        print(f"[RAG] 开始解析文档: {file_path}")

        try:
            raw_docs = self.load_document(file_path)
        except Exception as e:
            print("[RAG] 文档加载失败:", e)
            return []

        # 写入前，先删除相同 doc_id 的旧数据，避免重复
        try:
            self.vector_db.delete(where={"doc_id": doc_id})
            print(f"[RAG] 已删除 doc_id={doc_id} 的旧向量数据（如果存在）。")
        except Exception as e:  # noqa: E722
            print("[RAG] 删除旧向量数据时出错:", e)

        chunks = self.text_splitter.split_documents(raw_docs)

        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            if self.is_noise(chunk.page_content):
                # 丢弃垃圾 / 注入内容
                continue

            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["chunk_index"] = i
            if original_name:
                chunk.metadata["file_name"] = original_name

            filtered_chunks.append(chunk)

        if not filtered_chunks:
            print("[RAG] 导入报告：所有切片都被视为垃圾数据，未写入向量库。")
            return []

        self.vector_db.add_documents(filtered_chunks)

        print(
            f"[RAG] 文档 {doc_id} 已写入 {len(filtered_chunks)} 个有效 chunks "
            f"（原始切片数 {len(chunks)}）"
        )
        return [c.page_content for c in filtered_chunks]


# =========================
# IntelligentAssistant：Router + Retrieval + GPT 回答
# =========================

class IntelligentAssistant:
    def __init__(self, vector_db: Chroma, llm: ChatOpenAI):
        self.vector_db = vector_db
        self.llm = llm

        # 检索配置
        self.RECALL_K = 50               # 召回上限
        self.RERANK_TOP_K = 3            # 最终使用的 chunk 数
        # similarity_search_with_score 返回“距离”（越小越相似）
        self.SCORE_THRESHOLD = 999.0
        # 至少 15% 的有效词重叠才算“相关”
        self.MIN_OVERLAP_RATIO = 0.15

        # 使用全局停用词
        self.STOPWORDS = set(GLOBAL_STOPWORDS)

    # -------- Router（更“宽松”，默认倾向于检索） -------- #
    def _check_retrieval_necessity(self, query: str) -> bool:
        prompt = PromptTemplate.from_template(
            """You are a router of a Retrieval-Augmented Generation (RAG) system.

Your job:
- Decide whether the user's question needs to look up the uploaded textbook / documents.

If the question asks about:
- "this document", "this page", "the project", "the textbook", "the slides", or
- any specific facts that are likely stored in the uploaded materials (deadlines, requirements, architecture, etc.)

then you MUST answer YES.

If the question is pure casual chat or general chit-chat (e.g., "tell me a joke", "how are you"), answer NO.

Return ONLY one word: YES or NO.

Query: {query}
"""
        )
        result = (prompt | self.llm).invoke({"query": query})
        raw = result.content.strip().upper()
        print(f"[RAG] Router 原始输出：{raw}")

        has_yes = "YES" in raw
        has_no = "NO" in raw

        # 简单小聊天关键词判断
        small_talk_keywords = ["JOKE", "HOW ARE YOU", "你好", "笑话", "聊聊", "天气"]

        if any(kw in query.upper() for kw in small_talk_keywords):
            decision = False
        elif has_yes and not has_no:
            decision = True
        elif has_no and not has_yes:
            decision = False
        else:
            # 模型输出含糊 / 同时包含 YES 和 NO / 乱写 → 出于安全，默认执行检索
            decision = True

        print(f"[RAG] Router 判定（最终）：{'YES' if decision else 'NO'}")
        return decision

    # -------- 工具：计算“去停用词后的词面重叠比例” -------- #
    def _lexical_overlap(self, query: str, doc_text: str) -> float:
        def tokenize(text: str):
            # 简单按空格切 + 小写 + 去掉停用词
            tokens = set()
            for t in text.lower().split():
                # 去掉首尾标点（简单处理）
                t = t.strip(".,!?;:\"'()[]")
                if not t:
                    continue
                if t in self.STOPWORDS:
                    continue
                tokens.add(t)
            return tokens

        q_tokens = tokenize(query)
        d_tokens = tokenize(doc_text)

        if not q_tokens:
            return 0.0
        inter = q_tokens.intersection(d_tokens)
        return len(inter) / len(q_tokens)

    # -------- Retrieval + 基于“词面重叠”的知识盲区判断 -------- #
    def _retrieve(self, query: str) -> List[str]:
        """
        使用 Chroma 的 similarity_search_with_score：
        - 返回 (Document, score)，score 越小越相似（余弦距离）
        - 这里主要使用“去停用词后的词面重叠”来判断是否相关：
            - 重叠比例 >= MIN_OVERLAP_RATIO → 认为相关
            - 否则视为“知识盲区”噪声
        """
        print(f"[RAG] 开始向量检索 (k={self.RECALL_K}) ...")

        docs_and_scores = self.vector_db.similarity_search_with_score(
            query,
            k=self.RECALL_K,
        )

        if len(docs_and_scores) == 0:
            print("[RAG] 未召回到任何文档")
            return []

        scored = []
        for doc, score in docs_and_scores:
            overlap = self._lexical_overlap(query, doc.page_content)
            scored.append((doc, score, overlap))

        print("[RAG] 检索结果（前几条）的 score 和词面重叠（去停用词）：")
        for i, (doc, score, overlap) in enumerate(scored[:5]):
            print(f"  #{i+1}: score={score:.4f}, overlap={overlap:.4f}")

        # 依据词面重叠过滤无关内容
        filtered = [
            (doc, score, overlap)
            for doc, score, overlap in scored
            if overlap >= self.MIN_OVERLAP_RATIO
        ]

        if len(filtered) == 0:
            print("[RAG] 所有召回文档在词面上几乎没有有效重叠，判定为知识盲区。")
            return []

        # 对通过过滤的结果，按 score 从小到大排序，取前 RERANK_TOP_K 条
        filtered.sort(key=lambda x: x[1])
        top_docs = filtered[:self.RERANK_TOP_K]

        print(f"[RAG] 经过词面过滤后，剩余 {len(filtered)} 条；最终选用 {len(top_docs)} 条。")
        return [doc.page_content for doc, _, _ in top_docs]

    # -------- GPT-4o 最终回答（走代理） -------- #
    def _answer_with_gpt4o(self, query: str, chunks: List[str]) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL,  # 代理地址
        )

        if not chunks:
            # 知识盲区：没有足够信息
            return "根据已有教材内容，我无法找到与你的问题相关的知识，因此无法给出准确回答。"

        system_prompt = """
你是一名智能学习助手，你只能根据“提供的教材内容”回答问题。
必须满足以下规则：
1. 只能使用 chunks 中的信息，不得编造。
2. 如果 chunks 中并未包含足够信息，请回答：
   “根据已有教材内容，我无法找到与你的问题相关的知识。”
3. 回答必须清晰、简洁、可信赖。
"""

        user_content = f"用户问题：{query}\n\n以下是检索到的教材知识片段（chunks）：\n\n"
        for i, c in enumerate(chunks):
            user_content += f"[Chunk #{i+1}]\n{c}\n\n"

        completion = client.chat.completions.create(
            model=RAG_ANSWER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        return completion.choices[0].message.content or ""

    # -------- 总入口（给 handler / 测试调用） -------- #
    def handle_user_query(self, user_id: str, query: str, image_path=None):
        print(f"[RAG] 用户 {user_id} 提问：{query}")

        need_retrieval = self._check_retrieval_necessity(query)

        chunks: List[str] = []
        if need_retrieval:
            chunks = self._retrieve(query)

        final_answer = self._answer_with_gpt4o(query, chunks)

        return {
            "query": query,
            "retrieval_performed": need_retrieval,
            "matched_chunks": chunks,
            "final_answer": final_answer,
        }


# =========================
# 单例对象（供其他模块导入）
# =========================

kb_manager = KnowledgeBaseManager(vectorstore)
assistant = IntelligentAssistant(vectorstore, llm_router)

print("[RAG] RAG 服务初始化完成。")
