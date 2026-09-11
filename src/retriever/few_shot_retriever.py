"""模块 II：基于 RAG 的少样本检索器。
使用 TF-IDF + 余弦相似度，从训练集示例池中检索与当前样本最相似的 k 个已标注示例。
示例池仅来自训练划分（不含测试集），避免数据泄漏。
"""
import pandas as pd
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FewShotRetriever:
    def __init__(self, examples_path: str, k: int = 3, max_pool_size: int = 20000, random_state: int = 42):
        self.k = k
        df = pd.read_csv(examples_path)
        # 兼容列名：sentence 或 text
        if 'text' not in df.columns and 'sentence' in df.columns:
            df = df.rename(columns={'sentence': 'text'})
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label', 'text'])
        df['label'] = df['label'].astype(int)
        # 大池子抽样以控制向量化成本
        if len(df) > max_pool_size:
            df = df.sample(n=max_pool_size, random_state=random_state)
        self.pool = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.pool['text'].astype(str))
        print(f"FewShotRetriever built: pool={len(self.pool)} examples from {examples_path}")

    def retrieve(self, query_text: str, k: int = None) -> List[Dict[str, Any]]:
        """检索与查询文本最相似的 k 个示例，返回 [{text, label, score}]，按相似度降序。"""
        k = k if k is not None else self.k
        q = self.vectorizer.transform([str(query_text)])
        sims = cosine_similarity(q, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [
            {
                "text": self.pool.loc[i, 'text'],
                "label": int(self.pool.loc[i, 'label']),
                "score": float(sims[i]),
            }
            for i in top_idx
        ]
