import faiss
import numpy as np
import requests
from typing import List

class RAGSystem:
    def __init__(self, index_path: str):
        # 加载FAISS索引和元数据
        self.index = faiss.read_index(index_path)
        self.api_key = "sk-knaharxyaaloysxfdndezpdyluzywxhrkalqcjvjdyamgwup"
        
        # 加载元数据文件
        try:
            self.metadata = np.load("metadata.npy", allow_pickle=True)
        except FileNotFoundError:
            raise ValueError("未找到metadata.npy元数据文件")
            
        # 验证元数据长度
        if len(self.metadata) != self.index.ntotal:
            raise ValueError("元数据与索引数量不匹配")
            
        # 假设最大上下文长度
        self.max_context_length = 3800  # 为回答留出空间
        
    def get_embedding(self, text: str) -> np.ndarray:
        """调用BGE-M3模型获取文本嵌入"""
        url = "https://api.siliconflow.cn/v1/embeddings"
        payload = {
            "model": "Pro/BAAI/bge-m3",
            "encoding_format": "float",
            "input": text
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            embedding = np.array(response.json()['data'][0]['embedding'], dtype='float32')
            return embedding.reshape(1, -1)
        except Exception as e:
            print(f"嵌入获取失败: {str(e)}")
            return None

    def retrieve_context(self, query: str, k: int=3) -> List[str]:
        """从FAISS索引检索相关上下文"""
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
            
        # 执行相似性搜索
        distances, indices = self.index.search(query_embedding, k)
        
        # 将索引ID映射到元数据文本
        return [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]

    def generate_response(self, query: str, context: List[str]) -> str:
        """调用DeepSeek模型生成回答"""
        url = "https://api.siliconflow.cn/v1/chat/completions"
        
        # 构建提示词
        context_str = "\n".join(context)
        prompt = f"""基于以下小说内容，请以角色身份回答读者问题：
        
{context_str}

问题：{query}
回答："""
        
        # 截断过长的上下文
        if len(prompt) > self.max_context_length:
            prompt = prompt[:self.max_context_length] + "... [截断]"
            
        payload = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.5
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"回答生成失败：{str(e)}"

def main():
    # 初始化RAG系统
    rag = RAGSystem("journey_index.faiss")
    
    print("欢迎来到互动小说问答系统！输入'exit'退出")
    while True:
        query = input("\n你的问题：")
        if query.lower() in ['exit', 'quit']:
            break
            
        # 检索上下文
        context = rag.retrieve_context(query)
        if not context:
            print("未能检索到相关上下文")
            continue
            
        # 生成回答
        response = rag.generate_response(query, context)
        print("\n系统回答：")
        print(response.strip())

if __name__ == "__main__":
    main()
