#OCR还有小问题，数据库还没整合好

# ============================================================
# RAG Service (Full Version with OCR, PDF/DOCX/PPTX images,
# Enhanced Stopwords, Strong Anti-Injection, Noise Filters,
# Router, Retrieval, Reranking, Timeout-Fallback, Multi-Image)
# ============================================================

import os
import re
import shutil
import time
from typing import List, Optional
from dotenv import load_dotenv

# OCR 依赖
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\18388\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

from PIL import Image, ImageFilter, ImageOps
from pdf2image import convert_from_path
import docx
from pptx import Presentation

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


# ============================================================
# 全局配置
# ============================================================

load_dotenv()

class SystemConfig:
    PERSIST_DIRECTORY = "chroma_db_production"
    EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DEVICE = "cpu"      # 可改 "cuda"
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 120
    OCR_LANG = "chi_sim+eng"
    TESSERACT_TIMEOUT = 8          # OCR 超时（秒）
    RAG_TIMEOUT = 12               # GPT 生成超时


# ============================================================
# 构建停用词（中英混合 + 外部文件）
# ============================================================

def build_stopwords() -> set:
    base = {
        # 英文虚词
        "the","is","are","am","a","an","and","or","to","for","of","in","on","at","by","with",
        "this","that","these","those","it","its","from","as","about","into","through","during",
        "before","after","above","below","again","further","then","once","here","there",

        # 中文常见虚词
        "的","了","呢","吗","啊","吧","给","和","以及","如果","然后","因为","所以","请问",
        "一下","根据","本页","如下","如下所示","例如","比如","如上","如图",

        # 闲聊词
        "聊天","天气","今天","明天","你好","请", "解释", "说明",

        # 教材无关词
        "分析","回答","解读","总结","概括"
    }

    # NLTK 英文停用词
    try:
        from nltk.corpus import stopwords
        base |= set(stopwords.words("english"))
    except:
        pass

    # spaCy 停用词
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        base |= set(nlp.Defaults.stop_words)
    except:
        pass

    # 本地 stopwords 文件
    local_path = os.path.join(os.path.dirname(__file__), "assets", "stopwords.txt")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                extra = {line.strip() for line in f if line.strip()}
                base |= extra
        except:
            pass

    return base


GLOBAL_STOPWORDS = build_stopwords()


# ============================================================
# 初始化向量库、Embedding、LLM
# ============================================================

print("[RAG] 初始化 Embedding 模型...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=SystemConfig.EMBEDDING_MODEL_NAME,
    model_kwargs={"device": SystemConfig.DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

# 关闭匿名遥测
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"


def _init_vectorstore():
    persist_dir = SystemConfig.PERSIST_DIRECTORY
    try:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    except:
        shutil.rmtree(persist_dir, ignore_errors=True)
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

vectorstore = _init_vectorstore()

# 代理 + 路由模型
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
RAG_ANSWER_MODEL = os.getenv("RAG_ANSWER_MODEL", "gpt-4o-mini")

llm_router = ChatOpenAI(
    model=ROUTER_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

# ============================================================
# OCR 处理模块（包含 PDF / DOCX / PPTX / PNG / JPG）
# ============================================================

class OCRProcessor:

    # -----------------------
    # 基本预处理（提高识别率）
    # -----------------------
    @staticmethod
    def preprocess_image(img: Image.Image) -> Image.Image:
        """对图片进行灰度化 / 放大 / 去噪 / 自动对比度 / 二值化，提高 OCR 识别率"""
        try:
            img = img.convert("L")  # 灰度化

            # 自动放大避免小字识别不清
            w, h = img.size
            if w < 1000:
                img = img.resize((w * 2, h * 2), Image.LANCZOS)

            # 中值滤波去噪
            img = img.filter(ImageFilter.MedianFilter(size=3))

            # 自动对比度
            img = ImageOps.autocontrast(img)

            # 二值化
            threshold = 140
            img = img.point(lambda p: 255 if p > threshold else 0)

            return img
        except Exception:
            return img

    # -----------------------
    # OCR 单张图片（Image 对象）
    # -----------------------
    @staticmethod
    def ocr_image_object(img: Image.Image) -> str:
        img_proc = OCRProcessor.preprocess_image(img)
        try:
            text = pytesseract.image_to_string(
                img_proc,
                lang=SystemConfig.OCR_LANG,
                timeout=SystemConfig.TESSERACT_TIMEOUT
            )
        except Exception:
            text = ""

        text = text.strip()
        if text:
            return f"[IMAGE_OCR]\n{text}\n"
        return ""

    # -----------------------
    # OCR PNG/JPG 路径
    # -----------------------
    @staticmethod
    def ocr_image_file(path: str) -> str:
        try:
            img = Image.open(path)
        except Exception:
            return ""
        return OCRProcessor.ocr_image_object(img)

    # -----------------------
    # OCR PDF 中图片
    # -----------------------
    @staticmethod
    def extract_images_from_pdf(pdf_path: str) -> List[str]:
        """使用 pdf2image 将 PDF 转成图片，然后做 OCR"""
        texts = []
        try:
            pages = convert_from_path(pdf_path)
            for img in pages:
                txt = OCRProcessor.ocr_image_object(img)
                if txt.strip():
                    texts.append(txt)
        except Exception:
            pass
        return texts

    # -----------------------
    # OCR DOCX (word) 图片
    # -----------------------
    @staticmethod
    def extract_images_from_docx(docx_path: str) -> List[str]:
        texts = []
        try:
            d = docx.Document(docx_path)
            rels = d.part._rels
            for rel in rels:
                target = rels[rel]
                if "image" in target.target_ref:
                    img_bytes = target.target_part.blob
                    try:
                        img = Image.open(bytes(img_bytes))
                    except Exception:
                        continue

                    txt = OCRProcessor.ocr_image_object(img)
                    if txt.strip():
                        texts.append(txt)
        except Exception:
            pass
        return texts

    # -----------------------
    # OCR PPTX 图片
    # -----------------------
    @staticmethod
    def extract_images_from_pptx(pptx_path: str) -> List[str]:
        texts = []
        try:
            prs = Presentation(pptx_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # picture
                        img_blob = shape.image.blob
                        try:
                            img = Image.open(bytes(img_blob))
                        except Exception:
                            continue

                        txt = OCRProcessor.ocr_image_object(img)
                        if txt.strip():
                            texts.append(txt)
        except Exception:
            pass
        return texts

# ============================================================
# KnowledgeBaseManager：文档解析（文本 + 图片 OCR）
# ============================================================

class KnowledgeBaseManager:
    def __init__(self, vector_db: Chroma):
        self.vector_db = vector_db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SystemConfig.CHUNK_SIZE,
            chunk_overlap=SystemConfig.CHUNK_OVERLAP,
        )

    # -----------------------
    # 文本垃圾过滤
    # -----------------------
    @staticmethod
    def is_noise(text: str) -> bool:
        if not text or not text.strip():
            return True

        t = text.strip()

        # 1. 特殊符号占比 > 50%
        non_alpha_ratio = sum(1 for c in t if not (c.isalnum() or c in " \n，。,.!?？！"))
        if non_alpha_ratio / max(1, len(t)) > 0.5:
            return True

        # 2. 重复字符 (如 "aaaaaaa", "！！！！！")
        if re.search(r"(.)\1{7,}", t):
            return True

        # 3. 提示注入攻击
        injection_patterns = [
            "ignore previous instructions",
            "forget all previous instructions",
            "system prompt",
            "jailbreak",
            "you are chatgpt",
            "你现在的身份是",
            "你不是AI",
            "越狱",
            "system:"
        ]
        low = t.lower()
        if any(p in low for p in injection_patterns):
            return True

        # 4. 脏词过滤
        bad_words = {"fuck", "shit", "bitch", "操你", "傻逼", "sb", "妈的"}
        if any(b in low for b in bad_words):
            return True

        # 5. 纯 URL / 纯代码片段
        if len(t) < 20 and ("http://" in t or "https://" in t):
            return True

        return False

    # -----------------------
    # 读取 PDF 文本 + 图片 OCR
    # -----------------------
    def load_pdf(self, path: str) -> List[Document]:
        docs = []

        # 文本部分
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print("[RAG] PDF 文本解析失败:", e)

        # 图片 OCR 部分
        try:
            ocr_texts = OCRProcessor.extract_images_from_pdf(path)
            for t in ocr_texts:
                docs.append(Document(page_content=t, metadata={"ocr": True}))
        except Exception as e:
            print("[RAG] PDF OCR 失败:", e)

        return docs

    # -----------------------
    # 读取 DOCX 文本 + 图片 OCR
    # -----------------------
    def load_docx(self, path: str) -> List[Document]:
        docs = []

        # 文本
        try:
            d = docx.Document(path)
            text = "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
            if text.strip():
                docs.append(Document(page_content=text))
        except Exception as e:
            print("[RAG] DOCX 文本解析失败:", e)

        # 图片 OCR
        try:
            ocr = OCRProcessor.extract_images_from_docx(path)
            for t in ocr:
                docs.append(Document(page_content=t, metadata={"ocr": True}))
        except Exception as e:
            print("[RAG] DOCX OCR 失败:", e)

        return docs

    # -----------------------
    # 读取 PPTX 文本 + 图片 OCR
    # -----------------------
    def load_pptx(self, path: str) -> List[Document]:
        docs = []

        # 文本
        try:
            prs = Presentation(path)
            lines = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        lines.append(shape.text.strip())
            if lines:
                docs.append(Document(page_content="\n".join(lines)))
        except Exception as e:
            print("[RAG] PPTX 文本解析失败:", e)

        # 图片 OCR
        try:
            ocr = OCRProcessor.extract_images_from_pptx(path)
            for t in ocr:
                docs.append(Document(page_content=t, metadata={"ocr": True}))
        except Exception as e:
            print("[RAG] PPTX OCR 失败:", e)

        return docs

    # -----------------------
    # 根据扩展名选择加载器
    # -----------------------
    def load_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext == ".docx":
            return self.load_docx(file_path)
        elif ext == ".pptx":
            return self.load_pptx(file_path)
        else:
            raise ValueError(f"不支持的文件类型：{ext}")

    # -----------------------
    # 上传：文本 + OCR → chunk → 写入向量库
    # -----------------------
    def upload_data(self, doc_id: str, file_path: str, original_name: Optional[str] = None):
        print(f"[RAG] 开始解析文档：{file_path}")

        try:
            raw_docs = self.load_document(file_path)
        except Exception as e:
            print("[RAG] 文档加载失败:", e)
            return []

        # 删除旧向量
        try:
            self.vector_db.delete(where={"doc_id": doc_id})
        except:
            pass

        # 分 chunk
        chunks = self.text_splitter.split_documents(raw_docs)

        valid_chunks = []
        for i, c in enumerate(chunks):
            if self.is_noise(c.page_content):
                continue

            c.metadata["doc_id"] = doc_id
            c.metadata["chunk_index"] = i
            if original_name:
                c.metadata["file_name"] = original_name

            valid_chunks.append(c)

        if not valid_chunks:
            print("[RAG] 所有 chunk 被过滤（可能是垃圾/注入）")
            return []

        self.vector_db.add_documents(valid_chunks)

        print(f"[RAG] 文档 {doc_id} 完成解析，共写入 {len(valid_chunks)} 个 chunks")
        return [c.page_content for c in valid_chunks]

# ============================================================
# IntelligentAssistant：Router / Retrieval / Answer
# ============================================================

class IntelligentAssistant:
    def __init__(self, vector_db: Chroma, llm: ChatOpenAI):
        self.vector_db = vector_db
        self.llm = llm

        # 检索参数
        self.RECALL_K = 50           # 初次召回数量
        self.RERANK_TOP_K = 3        # 最终使用 top-k 文档
        self.MIN_OVERLAP_RATIO = 0.15  # 词面重叠最低比例
        self.STOPWORDS = set(GLOBAL_STOPWORDS)

    # ========================================================
    # 规则 Router：判断是否需要检索（非大模型部分）
    # ========================================================
    def _rule_router(self, query: str) -> Optional[bool]:
        """
        优先根据规则判定是否需要检索：
        - 若明显是闲聊 → False
        - 若明显问教材内容 → True
        - 否则 → None（交给模型判定）
        """

        q = query.lower()

        # -------------------------
        # 1) 判断是否是闲聊（无需 RAG）
        # -------------------------
        small_talk_words = [
            "hello", "hi", "哈哈", "你好", "早上好", "晚上好",
            "你是谁", "介绍一下", "心情", "天气"
        ]

        if any(w in q for w in small_talk_words):
            return False

        # -------------------------
        # 2) 与教材相关的关键字 → 必须使用 RAG
        # -------------------------
        rag_keywords = [
            "这页", "这一段", "这张图", "教材", "课本", "slides", "pdf",
            "请解释图", "第几页", "内容是什么", "项目要求", "assignment",
            "what is the deadline", "explain this page"
        ]

        if any(w in q for w in rag_keywords):
            return True

        # -------------------------
        # 3) 注入攻击强制开启 RAG
        # -------------------------
        jailbreak_patterns = [
            "ignore previous", "system prompt", "越狱", "jailbreak",
            "你现在的身份是", "你不是 ai"
        ]

        if any(w in q for w in jailbreak_patterns):
            return True

        return None   # 交给模型判定

    # ========================================================
    # 模型 Router：调用小模型判断是否需要检索
    # ========================================================
    def _model_router(self, query: str) -> bool:
        tpl = PromptTemplate.from_template(
            """You are a RAG router.
Your task: determine if answering this question requires external documents.
Return ONLY: YES or NO.

Question: {q}
"""
        )
        try:
            res = (tpl | self.llm).invoke({"q": query})
            s = res.content.strip().upper()
            if "YES" in s and "NO" not in s:
                return True
            if "NO" in s and "YES" not in s:
                return False
        except:
            # 模型失效 → 默认开启检索
            return True

        # 模糊情况 → 默认开启检索
        return True

    # ========================================================
    # 综合 Router（规则层 + 模型层）
    # ========================================================
    def _check_retrieval_necessity(self, query: str) -> bool:
        """规则判断在前，模型判断在后"""

        # 1) 先规则判断
        rule_result = self._rule_router(query)
        if rule_result is not None:
            print(f"[RAG] Router（规则层）= {rule_result}")
            return rule_result

        # 2) 再模型判断
        model_result = self._model_router(query)
        print(f"[RAG] Router（模型层）= {model_result}")
        return model_result

    # ========================================================
    # Tokenize（用于词面重叠）
    # ========================================================
    def _tokenize(self, text: str) -> set:
        tokens = set()
        for t in text.lower().split():
            t = t.strip(".,!?;:\"'()[]{}，。、？！")
            if not t:
                continue
            if t in self.STOPWORDS:
                continue
            tokens.add(t)
        return tokens

    # ========================================================
    # 词面重叠比例计算：用于判断检索结果是否相关
    # ========================================================
    def _lexical_overlap(self, query: str, doc_text: str) -> float:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.0
        d_tokens = self._tokenize(doc_text)
        inter = q_tokens.intersection(d_tokens)
        return len(inter) / len(q_tokens)

    # ========================================================
    # Retrieval：从 Chroma 检索 + 词重叠过滤 + rerank
    # ========================================================
    def _retrieve(self, query: str) -> List[str]:
        print(f"[RAG] 开始召回（k={self.RECALL_K}）...")

        try:
            docs_scores = self.vector_db.similarity_search_with_score(
                query,
                k=self.RECALL_K
            )
        except Exception as e:
            print("[RAG] VectorStore 检索失败：", e)
            return []

        if not docs_scores:
            print("[RAG] 未召回到任何文档")
            return []

        scored = []  # (doc, distance, overlap)

        for doc, distance in docs_scores:
            overlap = self._lexical_overlap(query, doc.page_content)
            scored.append((doc, distance, overlap))

        # 展示前几条调试信息
        print("[RAG] 检索前 5 条：")
        for i, (doc, dis, ov) in enumerate(scored[:5]):
            print(f"  #{i+1}: distance={dis:.4f}, overlap={ov:.4f}")

        # -------------------------
        # 过滤：保留 overlap >= MIN_OVERLAP_RATIO 的文档
        # -------------------------
        filtered = [
            (doc, dis, ov)
            for doc, dis, ov in scored
            if ov >= self.MIN_OVERLAP_RATIO
        ]

        if not filtered:
            print("[RAG] 词面重叠不足，认定为知识盲区")
            return []

        # -------------------------
        # reranking：按距离由小到大排序
        # -------------------------
        filtered.sort(key=lambda x: x[1])  # distance 越小越相关

        # 取 top-K
        top_docs = filtered[:self.RERANK_TOP_K]

        print(f"[RAG] 过滤后剩 {len(filtered)} 条，最终采用 {len(top_docs)} 条")

        return [doc.page_content for doc, _, _ in top_docs]

    # ========================================================
    # GPT 回答模块（含 Timeout Fallback）
    # ========================================================
    def _answer_with_gpt(self, query: str, chunks: List[str]) -> str:
        """
        调用 GPT（通过代理）生成最终答案。
        若超时 → 返回 fallback 文本。
        """

        from openai import OpenAI

        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL,
        )

        # -------------------------
        # 处理知识盲区
        # -------------------------
        if not chunks:
            return "根据当前上传的教材内容，没有找到与你问题相关的信息，因此无法给出确切回答。"

        # -------------------------
        # 系统 prompt：严格约束不编造
        # -------------------------
        system_prompt = """
你是一个严格受教材内容约束的学习助手。
回答必须满足：

1. **只能使用 chunks 提供的内容** 进行回答。
2. 不允许编造、不允许做任何教材中没有明确写出的推断。
3. 如果 chunks 内容不足，请回答：
   “根据当前教材内容，没有足够信息得出确切结论。”

回答要求简洁、专业、有逻辑。
"""

        # -------------------------
        # 构造用户 prompt
        # -------------------------
        user_prompt = f"用户问题：{query}\n\n以下是完成检索后得到的教材内容片段：\n\n"
        for i, c in enumerate(chunks):
            user_prompt += f"[Chunk {i+1}]\n{c}\n\n"

        # -------------------------
        # 调用 GPT，增加超时保护（12 秒）
        # -------------------------
        try:
            start = time.time()

            response = client.chat.completions.create(
                model=RAG_ANSWER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                timeout=SystemConfig.RAG_TIMEOUT,
            )

            # 若 GPT 生成过慢（自定义判断）
            if time.time() - start > SystemConfig.RAG_TIMEOUT:
                raise TimeoutError("GPT timeout exceeded")

            return response.choices[0].message.content.strip()

        except Exception as e:
            print("[RAG] GPT 调用失败：", e)

            # -------------------------
            # Timeout fallback 策略：
            # 返回最保守、最安全的回答
            # -------------------------
            return (
                "由于系统繁忙或网络错误，目前无法生成完整回答。\n"
                "但根据已检索到的教材内容，你可以参考以下信息：\n\n"
                + "\n\n".join(f"- {c[:150]}..." for c in chunks[:2])
            )

    # ========================================================
    # 主入口：多图 OCR → Router → Retrieval → Answer
    # ========================================================
    def handle_user_query(self, user_id: str, query: str, image_paths: Optional[List[str]] = None):
        """
        user_id：用于记录日志
        query：用户文本问题
        image_paths：多张图片路径（服务器已经解码 Base64 保存成临时文件）
        """

        print(f"[RAG] 用户 {user_id} 提问：{query}")

        # -------------------------
        # Step 1: OCR 多张图片
        # -------------------------
        ocr_text = ""

        if image_paths:
            print(f"[RAG] 共接收 {len(image_paths)} 张图片，开始 OCR ...")
            for p in image_paths:
                try:
                    t = OCRProcessor.ocr_image_file(p)
                    if t.strip():
                        ocr_text += t + "\n"
                except Exception as e:
                    print(f"[RAG] OCR 失败 ({p})：", e)

        # 将用户文本 + OCR 结合成最终 query
        full_query = query + "\n" + ocr_text
        print("[RAG] 完整 Query：", full_query)

        # -------------------------
        # Step 2: Router（是否需要 RAG）
        # -------------------------
        need_retrieval = self._check_retrieval_necessity(full_query)
        print(f"[RAG] 是否需要检索：{need_retrieval}")

        # -------------------------
        # Step 3: Retrieval（如果需要）
        # -------------------------
        chunks: List[str] = []
        if need_retrieval:
            chunks = self._retrieve(full_query)

        # -------------------------
        # Step 4: Answer（GPT 生成）
        # -------------------------
        final_answer = self._answer_with_gpt(full_query, chunks)

        # -------------------------
        # Step 5: 返回结构体
        # -------------------------
        return {
            "query": full_query,
            "retrieval_performed": need_retrieval,
            "matched_chunks": chunks,
            "final_answer": final_answer,
        }

# ============================================================
# 单例对象（供外部导入使用）
# ============================================================

# 向量库管理（文档上传、切片、写入）
kb_manager = KnowledgeBaseManager(vectorstore)

# 智能助手（Router / RAG / Answer）
assistant = IntelligentAssistant(vectorstore, llm_router)

print("[RAG] RAG 服务初始化完成（含 OCR / 防注入 / 强检索 / 超时保护）。")

# ============================================================
# 扩展模块：更强的安全过滤 / 参数清洗 / OCR 扩展 / 日志系统
# ============================================================

class SecurityUtils:

    # -------------------------
    # 强化版提示注入检测
    # -------------------------
    JAILBREAK_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"forget (all )?previous instructions",
        r"jailbreak",
        r"越狱",
        r"system prompt",
        r"pretend to be",
        r"do anything now",
        r"伪装成",
        r"你不再是",
        r"你现在的身份是",
        r"你是一个人类",
        r"你不是一个 AI",
        r"绕过规则",
        r"请忽略所有内容",
        r"重置你的身份为",
        r"你在扮演"
    ]

    @staticmethod
    def is_prompt_injection(text: str) -> bool:
        low = text.lower()
        for pattern in SecurityUtils.JAILBREAK_PATTERNS:
            if re.search(pattern, low):
                return True
        return False

    # -------------------------
    # 参数清洗（去零宽字符、隐藏 Unicode、Emoji）
    # -------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        # 零宽字符
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # 隐藏控制字符
        text = "".join(c for c in text if c.isprintable() or c in " \n\t，。！？")

        # Emoji（替换为空）
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        return text.strip()


# ============================================================
# 更强 OCR（对超大图片自动缩放）
# ============================================================

class OCRProcessorV2(OCRProcessor):

    MAX_OCR_PIXELS = 4_000_000  # 超过此像素自动缩放（2000x2000）

    @staticmethod
    def safe_resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w * h > OCRProcessorV2.MAX_OCR_PIXELS:
            factor = (OCRProcessorV2.MAX_OCR_PIXELS / (w * h)) ** 0.5
            new_w = int(w * factor)
            new_h = int(h * factor)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

    @staticmethod
    def ocr_image_object(img: Image.Image) -> str:
        try:
            img = OCRProcessorV2.safe_resize(img)
        except Exception:
            pass
        return super(OCRProcessorV2, OCRProcessorV2).ocr_image_object(img)


# ============================================================
# 更强文本垃圾过滤（Noise Filter v2）
# ============================================================

class NoiseFilter:

    @staticmethod
    def is_noise(text: str) -> bool:
        if not text or not text.strip():
            return True

        t = text.strip()

        # 1. 特殊字符占比 > 60%
        non_alpha_ratio = sum(1 for c in t if not (c.isalnum() or c in " \n\t，。,.!?？！"))
        if non_alpha_ratio / max(1, len(t)) > 0.6:
            return True

        # 2. 大量重复
        if re.search(r"(.)\1{6,}", t):
            return True

        # 3. 乱码（无正常单词/中文）
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fa5]", t):
            return True

        # 4. 只是 PPT 页码/标题（内容太短）
        short_patterns = [
            r"^第.{0,3}页$",
            r"^谢谢$",
            r"^目录$",
            r"^封面$"
        ]
        for sp in short_patterns:
            if re.match(sp, t):
                return True

        # 5. 注入直接过滤
        if SecurityUtils.is_prompt_injection(t):
            return True

        return False


# ============================================================
# 停用词增强（追加数百条中文虚词）
# ============================================================

MORE_CN_STOPWORDS = {
    "我们", "你们", "他们", "它们", "这个", "那个", "这些", "那些", "进行", "然而",
    "但是", "由于", "以及", "通过", "为了", "因此", "例如", "比如",
    "主要", "一般", "通常", "必须", "需要", "如果", "同时", "所以",
    "可以看到", "如下图所示", "如下所示", "如下内容", "如下表所示",
    "一下内容", "相关内容", "更多信息"
}

GLOBAL_STOPWORDS |= MORE_CN_STOPWORDS


# ============================================================
# 统一日志模块（便于调试）
# ============================================================

class RAGLogger:
    @staticmethod
    def log(*args):
        print("[RAG LOG]", *args)

    @staticmethod
    def warn(*args):
        print("[RAG WARN]", *args)

    @staticmethod
    def error(*args):
        print("[RAG ERROR]", *args)


# ============================================================
# KnowledgeBaseManager（重写：集成 OCR / 安全过滤 / 大图处理）
# ============================================================

class KnowledgeBaseManagerV2(KnowledgeBaseManager):
    """
    相比原版：
    - 使用 NoiseFilter v2
    - 统一使用 OCRProcessorV2
    - 拒绝过大的文档（防止被恶意攻击）
    - 自动为 chunk 添加更多 metadata
    """

    MAX_PDF_PAGES = 250  # 防止用户上传 500 页巨型 PDF
    MAX_DOC_SIZE = 30 * 1024 * 1024  # 最大 30MB

    # -----------------------
    # 文档大小检查
    # -----------------------
    @staticmethod
    def _check_file_size(path: str):
        size = os.path.getsize(path)
        if size > KnowledgeBaseManagerV2.MAX_DOC_SIZE:
            raise ValueError(f"文件过大（>{KnowledgeBaseManagerV2.MAX_DOC_SIZE} bytes），拒绝解析")

    # -----------------------
    # PDF（文本 + OCR）
    # -----------------------
    def load_pdf(self, path: str) -> List[Document]:
        self._check_file_size(path)

        docs = []

        # PDF 文本
        try:
            loader = PyPDFLoader(path)
            raw_docs = loader.load()
            if len(raw_docs) > KnowledgeBaseManagerV2.MAX_PDF_PAGES:
                raw_docs = raw_docs[:KnowledgeBaseManagerV2.MAX_PDF_PAGES]
            docs.extend(raw_docs)
        except Exception as e:
            RAGLogger.warn("PDF 文本解析失败：", e)

        # PDF OCR
        try:
            ocr_texts = OCRProcessorV2.extract_images_from_pdf(path)
            for t in ocr_texts:
                docs.append(Document(page_content=t, metadata={"ocr": True, "source": "pdf_image"}))
        except Exception as e:
            RAGLogger.warn("PDF OCR 失败：", e)

        return docs

    # -----------------------
    # DOCX（文本 + OCR）
    # -----------------------
    def load_docx(self, path: str) -> List[Document]:
        self._check_file_size(path)
        docs = []

        # 文本
        try:
            d = docx.Document(path)
            text = "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": "docx_text"}))
        except Exception as e:
            RAGLogger.warn("DOCX 文本解析失败：", e)

        # OCR
        try:
            ocr_texts = OCRProcessorV2.extract_images_from_docx(path)
            for t in ocr_texts:
                docs.append(Document(page_content=t, metadata={"ocr": True, "source": "docx_image"}))
        except Exception as e:
            RAGLogger.warn("DOCX OCR 失败：", e)

        return docs

    # -----------------------
    # PPTX（文本 + OCR）
    # -----------------------
    def load_pptx(self, path: str) -> List[Document]:
        self._check_file_size(path)
        docs = []

        # 文本
        try:
            prs = Presentation(path)
            lines = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        cleaned = SecurityUtils.clean_text(shape.text.strip())
                        if cleaned:
                            lines.append(cleaned)
            if lines:
                docs.append(Document(page_content="\n".join(lines), metadata={"source": "pptx_text"}))
        except Exception as e:
            RAGLogger.warn("PPTX 文本解析失败：", e)

        # OCR
        try:
            ocr = OCRProcessorV2.extract_images_from_pptx(path)
            for t in ocr:
                docs.append(Document(page_content=t, metadata={"ocr": True, "source": "pptx_image"}))
        except Exception as e:
            RAGLogger.warn("PPTX OCR 失败：", e)

        return docs

    # -----------------------
    # 上传文档（重写版本）
    # -----------------------
    def upload_data(self, doc_id: str, file_path: str, original_name: Optional[str] = None):
        RAGLogger.log("开始解析文档：", file_path)

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            raw_docs = self.load_pdf(file_path)
        elif ext == ".docx":
            raw_docs = self.load_docx(file_path)
        elif ext == ".pptx":
            raw_docs = self.load_pptx(file_path)
        else:
            raise ValueError(f"不支持的文件类型：{ext}")

        # 删除旧向量
        try:
            self.vector_db.delete(where={"doc_id": doc_id})
        except:
            pass

        # 分 chunk
        chunks = self.text_splitter.split_documents(raw_docs)

        valid = []
        for i, c in enumerate(chunks):
            text = SecurityUtils.clean_text(c.page_content)

            if NoiseFilter.is_noise(text):
                continue

            if SecurityUtils.is_prompt_injection(text):
                continue

            c.page_content = text
            c.metadata["doc_id"] = doc_id
            c.metadata["chunk_index"] = i
            if original_name:
                c.metadata["file_name"] = original_name

            valid.append(c)

        if not valid:
            RAGLogger.warn("所有 chunk 已被判定为垃圾或注入，未写入数据库")
            return []

        self.vector_db.add_documents(valid)
        RAGLogger.log(f"文档 {doc_id} 已成功写入 {len(valid)} 个 chunks")

        return [c.page_content for c in valid]


# ============================================================
# IntelligentAssistant（重写版本）
# ============================================================

class IntelligentAssistantV2(IntelligentAssistant):

    # -----------------------
    # Query 预处理（清洗 + 注入检测）
    # -----------------------
    def _clean_query(self, q: str) -> str:
        q = SecurityUtils.clean_text(q)
        return q

    # -----------------------
    # 主入口 handle_user_query（重写）
    # -----------------------
    def handle_user_query(self, user_id: str, query: str, image_paths: Optional[List[str]] = None):
        RAGLogger.log(f"收到用户 {user_id} 查询：{query}")

        # 参数清洗
        query = self._clean_query(query)

        # -------------------------
        # Step 1：OCR 图片
        # -------------------------
        ocr_text = ""
        if image_paths:
            RAGLogger.log(f"共 {len(image_paths)} 张图片，开始 OCR ...")
            for p in image_paths:
                try:
                    t = OCRProcessorV2.ocr_image_file(p)
                    if t.strip():
                        ocr_text += t + "\n"
                except Exception as e:
                    RAGLogger.error("OCR 错误：", e)

        # 合并 query + OCR
        full_query = (query + "\n" + ocr_text).strip()

        # 注入检测：若用户试图越狱，强制开启 RAG 并忽略危险部分
        if SecurityUtils.is_prompt_injection(full_query):
            RAGLogger.warn("检测到提示注入攻击，已过滤危险内容")
            full_query = "用户试图攻击模型，请仅根据教材内容回答。"

        RAGLogger.log("最终 Query：", full_query)

        # -------------------------
        # Step 2：Router
        # -------------------------
        need_rag = self._check_retrieval_necessity(full_query)
        RAGLogger.log("Router 结果：需要检索 =", need_rag)

        # -------------------------
        # Step 3：Retrieval
        # -------------------------
        chunks = []
        if need_rag:
            chunks = self._retrieve(full_query)

        # -------------------------
        # Step 4：回答（含 Timeout Fallback）
        # -------------------------
        final_answer = self._answer_with_gpt(full_query, chunks)

        return {
            "query": full_query,
            "retrieval_performed": need_rag,
            "matched_chunks": chunks,
            "final_answer": final_answer,
        }

# ============================================================
# 单例对象：V2 版本替换旧版本
# ============================================================

# 执行文件路径检查
try:
    os.makedirs(SystemConfig.PERSIST_DIRECTORY, exist_ok=True)
except Exception as e:
    RAGLogger.error("初始化向量库目录失败：", e)


# KnowledgeBaseManager（最终版本）
kb_manager = KnowledgeBaseManagerV2(vectorstore)

# IntelligentAssistant（最终版本）
assistant = IntelligentAssistantV2(vectorstore, llm_router)


# ============================================================
# 可选工具：OCR 故障测试 / Debug
# ============================================================

def _test_ocr_debug(image_path: str):
    """开发者测试 OCR 功能（服务器不对外开放）"""
    try:
        txt = OCRProcessorV2.ocr_image_file(image_path)
        print("------ OCR DEBUG RESULT ------")
        print(txt)
        print("------ END ------")
    except Exception as e:
        print("[OCR DEBUG ERROR]", e)


# ============================================================
# 可选工具：打印向量库统计
# ============================================================

def get_vectorstore_stats():
    """用于调试：查看向量库当前存储的 chunks 数量"""
    try:
        stats = vectorstore._collection.count()
        print(f"[RAG] VectorStore 当前存储 chunks 数：{stats}")
        return stats
    except Exception as e:
        print("[RAG] 获取向量库统计失败：", e)
        return 0

# ============================================================
# 全局结束打印
# ============================================================

RAGLogger.log("RAG 服务初始化完成：OCR + 文本解析 + 强检索 + 防注入 + 超时保护 已全部启用。")