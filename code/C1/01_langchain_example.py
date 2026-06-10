import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
# 得到的respond：
# 根据提供的上下文，文中举了以下例子：

# 1. 自然界中，羚羊通过试错学习站立和奔跑。
# 2. 股票交易，通过买卖股票并根据市场反馈来最大化奖励。
# 3. 玩雅达利游戏（如Breakout和Pong）或其他电脑游戏，通过不断试错来学习如何通关。
# 4. 选择餐馆的例子，用于说明探索和利用的概念。
# 5. 做广告的例子，用于说明探索和利用的概念。
# 6. 挖油的例子，用于说明探索和利用的概念。
# 7. 玩《街头霸王》游戏的例子，用于说明探索和利用的概念。

# RecursiveCharacterTextSplitter的默认参数chunk_size=4000,chunk_overlap=200,
text_splitter = RecursiveCharacterTextSplitter()

# chunk_size=500, chunk_overlap=200 时，得到的回复只有
# 根据提供的上下文，文中举的例子有：选择餐馆、做广告、挖油、玩游戏。
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
# llm = ChatOpenAI(
#     model="glm-4.7-flash-free",
#     temperature=0.7,
#     max_tokens=4096,
#     # api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_key=os.getenv("AIHUBMIX_API_KEY"),

#     base_url="https://aihubmix.com/v1"
# )

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)
# print(answer)



