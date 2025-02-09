import requests
import json
import numpy as np
import faiss
from typing import List, Tuple

class NovelQA:
    def __init__(self):
        self.api_url = "https://api.siliconflow.cn/v1"
        self.headers = {
            "Authorization": "Bearer <your apikey>",
            "Content-Type": "application/json"
        }
        self.index = None
        self.text_chunks = []
        
    def chunk_text(self, file_path: str, chunk_size: int = 2048) -> List[str]:
        """按固定长度分割文本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    def get_embedding(self, text: str) -> List[float]:
        """调用BGE-M3模型获取嵌入"""
        payload = {
            "model": "Pro/BAAI/bge-m3",
            "input": text
        }
        response = requests.post(f"{self.api_url}/embeddings", 
                               json=payload, 
                               headers=self.headers)
        return json.loads(response.text)['data'][0]['embedding']
    
    def build_index(self, chunks: List[str]):
        """构建FAISS向量索引"""
        from tqdm import tqdm
        
        print("\n正在生成文本嵌入...")
        embeddings = []
        for chunk in tqdm(chunks, desc="处理进度", unit="chunk"):
            embeddings.append(self.get_embedding(chunk))
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.text_chunks = chunks
        faiss.write_index(self.index, "novel_embeddings.faiss")
    
    def search_chunks(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        """检索相关文本块"""
        query_embedding = self.get_embedding(query)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k)
        return [(distances[0][i], self.text_chunks[indices[0][i]]) 
               for i in range(k)]
    
    def generate_answer(self, context: str, question: str) -> str:
        """调用DeepSeek模型生成回答"""
        prompt = f"基于以下小说内容：\n{context}\n\n请回答：{question}"
        payload = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512
        }
        response = requests.post(f"{self.api_url}/chat/completions",
                               json=payload,
                               headers=self.headers)
        return json.loads(response.text)['choices'][0]['message']['content']
    
    def run(self):
        """运行交互式问答系统"""
        # 初始化索引
        chunks = self.chunk_text("WEST.txt")
        self.build_index(chunks)
        
        # 交互循环
        print("小说问答系统已启动（输入exit退出）")
        while True:
            question = input("\n请输入关于小说的问题：")
            if question.lower() == 'exit':
                break
                
            # 检索相关段落
            results = self.search_chunks(question)
            context = "\n".join([chunk for _, chunk in results])
            
            # 生成回答
            answer = self.generate_answer(context, question)
            print("\nAI回答：")
            print(answer.strip())

if __name__ == "__main__":
    qa = NovelQA()
    qa.run()
