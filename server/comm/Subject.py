import json
import jieba
import numpy as np
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib  # 用于模型保存和加载
from Parser import Parser  # 确保 Parser 类已实现 parse_file 方法


class SubjectModel:
    def __init__(self, threshold=0.05):
        """
        初始化学科分类模型（增强公式识别）
        :param threshold: 判定为'其他'的概率阈值
        """
        self.threshold = threshold
        self.is_trained = False

        # 扩展停用词表（保留公式相关词）
        self.stop_words = {
            '的', '了', '是', '在', '和', '跟', '与', '之', '吗', '呢', '个', '这', '那',
            'the', 'a', 'an', 'is', 'are', 'of', 'in', 'on', 'at', 'to', 'for', 'and',
            'please', 'help', 'me', 'find', 'calculate', 'what', 'how', 'why',
            '请', '帮我', '计算', '求解', '为什么', '怎么', '题目', '内容', '下列', '选项',
            '已知', '求', '如图', '所示'
        }

        # TF-IDF + Naive Bayes Pipeline
        self.model = make_pipeline(
            TfidfVectorizer(
                tokenizer=self._tokenizer,
                token_pattern=None,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents=None
            ),
            MultinomialNB(alpha=0.01)
        )
        self.classes_ = []

        # 学科模板字典，用于生成训练样本
        self.subject_templates = {
            "物理学": ["应用{kw}定律", "{kw}的物理意义", "实验验证{kw}", "计算{kw}的值", "推导{kw}的公式"],
            "应用物理学": ["{kw}实验分析", "{kw}的工程应用", "模拟{kw}", "测量{kw}参数", "优化{kw}"],
            "化学": ["{kw}反应方程式", "{kw}的化学计量", "合成{kw}", "分析{kw}", "{kw}的制备"],
            "生物科学": ["检测{kw}", "{kw}的表达", "{kw}的功能", "敲除{kw}", "过表达{kw}"],
            "生物技术": ["应用{kw}", "工程化{kw}", "生产{kw}", "优化{kw}", "{kw}改造"],
            "生物信息学": ["分析{kw}", "序列比对{kw}", "建模{kw}", "挖掘{kw}", "{kw}算法"],
            "生物医学工程": ["测量{kw}", "{kw}设备设计", "模拟{kw}", "优化{kw}", "实验{kw}"],
            "通信工程": ["设计{kw}协议", "实现{kw}系统", "优化{kw}", "仿真{kw}", "传输{kw}分析"],
            "光电信息科学与工程": ["测量{kw}", "仿真{kw}", "优化{kw}", "设计{kw}系统", "实验{kw}"],
            "信息工程": ["实现{kw}算法", "系统分析{kw}", "优化{kw}", "应用{kw}", "建模{kw}"],
            "材料科学与工程": ["制备{kw}", "表征{kw}", "分析{kw}", "优化{kw}", "性能测试{kw}"],
            "水文与水资源工程": ["模拟{kw}", "测量{kw}", "分析{kw}", "优化{kw}", "评价{kw}"],
            "环境科学与工程": ["监测{kw}", "模拟{kw}", "分析{kw}", "治理{kw}", "优化{kw}"],
            "计算机科学与技术": ["实现{kw}算法", "{kw}的时间复杂度", "优化{kw}", "应用{kw}", "训练{kw}模型"],
            "智能科学与技术": ["实现{kw}算法", "优化{kw}", "应用{kw}", "模型训练{kw}", "分析{kw}"],
            "金融数学": ["计算{kw}", "建模{kw}", "优化{kw}", "风险评估{kw}", "概率分析{kw}"],
            "数学与应用数学": ["证明{kw}", "计算{kw}", "解方程{kw}", "推导{kw}", "分析{kw}"],
            "统计学": ["计算概率P({kw})", "统计{kw}", "分析{kw}", "建模{kw}", "回归{kw}"],
            "理论与应用力学": ["分析{kw}", "模拟{kw}", "计算{kw}", "优化{kw}", "实验{kw}"],
            "航空航天工程": ["设计{kw}", "优化{kw}", "仿真{kw}", "测量{kw}", "实验{kw}"],
            "机械工程": ["设计{kw}", "分析{kw}", "优化{kw}", "制造{kw}", "测试{kw}"],
            "机器人工程": ["实现{kw}", "优化{kw}", "控制{kw}", "仿真{kw}", "实验{kw}"],
            "金融学": ["分析{kw}", "建模{kw}", "风险评估{kw}", "投资{kw}", "优化{kw}"],
            "金融工程": ["定价{kw}", "建模{kw}", "优化{kw}", "风险控制{kw}", "分析{kw}"],
            "海洋科学": ["测量{kw}", "分析{kw}", "模拟{kw}", "建模{kw}", "评价{kw}"],
            "地球物理学": ["分析{kw}", "建模{kw}", "测量{kw}", "模拟{kw}", "推导{kw}"],
            "生物医学科学": ["检测{kw}", "分析{kw}", "实验{kw}", "功能研究{kw}", "应用{kw}"],
            "临床医学": ["诊断{kw}", "治疗{kw}", "评估{kw}", "分析{kw}", "实验{kw}"],
            "微电子科学与工程": ["设计{kw}", "优化{kw}", "测量{kw}", "仿真{kw}", "制备{kw}"],
            "大数据管理与应用": ["分析{kw}", "存储{kw}", "处理{kw}", "优化{kw}", "建模{kw}"]
        }


    def _tokenizer(self, text):
        """增强版分词器：保留公式、化学式、数学符号、希腊字母、函数复杂度符号"""
        patterns = {
            r'\$(.*?)\$': 'LATEX',
            r'\\begin\{equation\}(.*?)\\end\{equation\}': 'LATEX',
            r'\\\[.*?\\\]': 'LATEX',
            r'\\begin\{align\}(.*?)\\end\{align\}': 'LATEX',
            r'([A-Z][a-z]?)(\d*)': 'CHEM',
            r'→|⇌|↑|↓|Δ': 'CHEM',
            r'∫|∑|∏|→|⇒|∀|∃|∈|⊂|⊆|∩|∪|∅|∞|∂|∇|≈|≠|≤|≥|±|×|÷': 'MATH',
            r'[α-ωΑ-Ω]': 'GREEK',
            r'[a-zA-Z]\(.*?\)': 'FUNC',
            r'O\(.*?\)': 'COMPLEX'
        }
        for pattern, tag in patterns.items():
            text = re.sub(pattern, lambda m: f' {tag}_{m.group(0)}_{tag} ', text, flags=re.DOTALL)

        tokens = jieba.lcut(text)
        clean_tokens = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            for tag in ['LATEX', 'CHEM', 'MATH', 'GREEK', 'FUNC', 'COMPLEX']:
                if t.startswith(f'{tag}_') and t.endswith(f'_{tag}'):
                    t = t[len(tag)+1:-len(tag)-1]
                    break
            if t.lower() in self.stop_words:
                continue
            clean_tokens.append(t.lower())
        return clean_tokens

    def _generate_synthetic_data(self, keyword, label):
        """生成增强训练样本"""
        samples = [keyword, f"关于{keyword}的题目", f"已知{keyword}的条件", f"求{keyword}的值", f"计算{keyword}", f"推导{keyword}", f"证明{keyword}", f"calculate the {keyword}", f"derive {keyword}", f"prove {keyword}"]

        if label in self.subject_templates:
            for tmpl in self.subject_templates[label]:
                samples.append(tmpl.format(kw=keyword))

        if any(c in keyword for c in "=→∫∑αβγδε∞∂∇"):
            samples.append(f"{keyword}的推导过程")
            samples.append(f"{keyword}的应用")
            samples.append(f"使用{keyword}解题")
            samples.append(f"{keyword}的证明")
        return samples

    def train(self, json_path):
        """从 JSON 文件加载关键词训练模型"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到关键词文件: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        texts, labels = [], []
        for subject, keywords in data_dict.items():
            for kw in keywords:
                samples = self._generate_synthetic_data(kw, subject)
                texts.extend(samples)
                labels.extend([subject]*len(samples))

        if not texts:
            raise ValueError("训练数据为空，请检查 JSON 文件内容。")

        print(f"构建了 {len(texts)} 条训练样本，开始训练模型...")
        self.model.fit(texts, labels)
        self.classes_ = self.model.classes_
        self.is_trained = True
        print("模型训练完成。")

    def predict(self, text):
        """预测文本学科"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 train() 方法。")
        if not text or not text.strip():
            return "其他"
        try:
            probas = self.model.predict_proba([text])[0]
            max_index = np.argmax(probas)
            max_proba = probas[max_index]
            if max_proba < self.threshold:
                return "其他"
            vector = self.model.steps[0][1].transform([text])
            if vector.nnz == 0:
                return "其他"
            return self.classes_[max_index]
        except Exception as e:
            print(f"预测出错: {e}")
            return "其他"

    def predict_with_proba(self, text, top_k=3):
        """返回 top_k 类别及概率"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 train() 方法。")
        if not text or not text.strip():
            return [("其他", 1.0)]
        try:
            probas = self.model.predict_proba([text])[0]
            top_indices = np.argsort(probas)[-top_k:][::-1]
            results = [(self.classes_[i], probas[i]) for i in top_indices if probas[i] >= self.threshold]
            return results if results else [("其他", 1.0)]
        except Exception as e:
            print(f"预测出错: {e}")
            return [("其他", 1.0)]

    def save(self, model_path="subject_model.joblib"):
        """保存模型"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法保存")
        model_data = {
            'model': self.model,
            'classes': self.classes_,
            'threshold': self.threshold,
            'version': 'v1.0',
            'trained_date': datetime.now().isoformat()
        }
        joblib.dump(model_data, model_path)
        print(f"✅ 模型已保存至: {model_path}")

    def load(self, model_path="subject_model.joblib"):
        """加载已保存模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.classes_ = model_data['classes']
        self.threshold = model_data['threshold']
        self.is_trained = True
        print(f"✅ 模型已从 {model_path} 加载")


# 使用示例
if __name__ == "__main__":
    clf = SubjectModel(threshold=0.03)
    parser = Parser()

    model_file = "subject_model.joblib"
    if os.path.exists(model_file):
        print("检测到已有模型文件，正在加载...")
        clf.load(model_file)
    else:
        print("未找到模型文件，开始训练新模型...")
        clf.train("keywords.json")
        clf.save(model_file)

    try:
        parsed_text = parser.parse_file("TWIST.pdf")["text"]  # 测试文件
        result = clf.predict(parsed_text)
        print(f"预测结果: {result}")
    except Exception as e:
        print(f"预测过程中出错: {e}")
