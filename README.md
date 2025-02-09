# RAG代码须知
## 源代码里面def __init__(self):
        self.api_url = "https://api.siliconflow.cn/v1"（这个地方是你用的模型的URL）
        self.headers = {
            "Authorization": "Bearer <your apikey>",（这个地方填你的apikey）
            "Content-Type": "application/json"
        }
        self.index = None
        self.text_chunks = []
## 由于本作者使用的是先调用硅基流动里面的Pro/BAAI/bge-m3模型进行文本embeddings生成，然后再用硅基流动的Pro/deepseek-ai/DeepSeek-R1模型进行输出答案，所以URL和apikey两个是一样的，然后硅基流动使用的是ChatGPT给的模版进行api调用，所以使用时两个模型首先要来自一个apikey，其次是用openai的接口。
## 本作者使用的faiss库是cpu，因为作者用的是Windows操作系统，没有faiss-gpu使用，所以加载过程会慢一点
## 本作者使用的文档是西游记，进行文本切割不是按章节来的而是用字数
# 优化建议
## 本作者代码只是初步的，所以第二次打开还会重复处理已经生成过的faiss索引库，可以写一个if来跳过
## 由于rag技术本身就有缺陷，它相当于是根据用户问题和通过faiss的一个算法匹配到最符合的几个chunk（此时是embedding形式），然后把chunk转换成文本形式直接发给ai生成提示词，所以像涉及到很多chunk文本内容的离散问题就回答的很差，我的想法是可以在生成提示词前分别把不同的chunk和问题单独抛给ai然后生成一个新的提示词，再把这些提示词合并发给ai笼统的给出更加精确的答案。（比如说统计具体的问题就很差）
# 作者是山东大学（威海）23信计的学生，目前第一天接触这个rag技术，如果有问题或者是想交流，欢迎加本人vx：liucong233333（菌子）
