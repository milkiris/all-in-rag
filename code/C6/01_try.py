import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 解决 HuggingFace 下载卡死问题

import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)
from llama_index.core.evaluation import DatasetGenerator # 更新了这里
from llama_index.core.llama_dataset import LabelledRagDataset

Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

async def main():
    # 1. 加载文档
    print("正在加载 PDF 文档...")
    reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
    documents = reader.load_data()
    
    # 为了测试能跑通，临时只取前 10 页进行后续的索引构建（测试成功后再去掉限制）
    test_documents = documents[:10] 

    # 1.1 加载或生成响应评估数据集
    dataset_path = "./c6_response_eval_dataset.json"
    if os.path.exists(dataset_path):
        print("加载响应评估数据集...")
        response_eval_dataset = LabelledRagDataset.from_json(dataset_path) # 使用新 API
    else:
        print("生成响应评估数据集 (这可能需要调用 LLM，请稍候)...")
        dataset_generator = DatasetGenerator.from_documents(test_documents)  
        response_eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=15)  
        response_eval_dataset.save_json(dataset_path)

    # 2. 构建两种不同的RAG查询引擎和检索器进行对比
    print("正在处理文档节点并构建句子窗口索引 (首次运行会下载 Embedding 模型，请耐心等待)...")
    # 2.1 句子窗口检索
    sentence_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    # 注意这里改为了 test_documents
    sentence_nodes = sentence_parser.get_nodes_from_documents(test_documents) 
    sentence_index = VectorStoreIndex(sentence_nodes)

    sentence_query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    print("正在构建常规分块索引...")
    # 2.2 常规分块检索（基准）
    base_parser = SentenceSplitter(chunk_size=512)
    # 注意这里改为了 test_documents
    base_nodes = base_parser.get_nodes_from_documents(test_documents)
    base_index = VectorStoreIndex(base_nodes)

    base_query_engine = base_index.as_query_engine(similarity_top_k=2)

    # 3. 初始化响应评估器
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

    # 4. 执行响应评估对比
    print("\n开始执行响应评估对比 (调用 DeepSeek API 进行评估)...")
    evaluators = {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator}
    
    # LlamaIndex 新版 Dataset 的 query 提取方式
    # 兼容处理：提取查询列表 (不论生成的是新版还是旧版数据集对象)
    if hasattr(response_eval_dataset, "examples"):
        # 适配新版 LabelledRagDataset
        queries = [example.query for example in response_eval_dataset.examples]
    else:
        # 适配旧版 QueryResponseDataset (它的 queries 是一个 {id: query_text} 的字典)
        queries = list(response_eval_dataset.queries.values())

    # 如果并发为 2 时 DeepSeek 报错 Rate Limit，可以尝试将 workers 改为 1
    print("\n=== 评估句子窗口检索 ===")
    sentence_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
    sentence_response_results = await sentence_runner.aevaluate_queries(
        queries=queries, query_engine=sentence_query_engine
    )

    print("\n=== 评估常规分块检索 ===")
    base_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
    base_response_results = await base_runner.aevaluate_queries(
        queries=queries, query_engine=base_query_engine
    )

    # 5. 分析并打印对比结果
    print("\n" + "="*60)
    print("响应评估结果对比")
    print("="*60)

    def calc_response_score(results, metric):
        if results and results.get(metric):
            scores = results[metric]
            return sum(r.passing for r in scores) / len(scores)
        return 0

    sentence_faith = calc_response_score(sentence_response_results, "faithfulness")
    sentence_rel = calc_response_score(sentence_response_results, "relevancy")

    base_faith = calc_response_score(base_response_results, "faithfulness")
    base_rel = calc_response_score(base_response_results, "relevancy")

    print(f"\n句子窗口检索:")
    print(f"  忠实度: {sentence_faith:.1%}")
    print(f"  相关性: {sentence_rel:.1%}")

    print(f"\n常规分块检索:")
    print(f"  忠实度: {base_faith:.1%}")
    print(f"  相关性: {base_rel:.1%}")

    if sentence_faith > base_faith and sentence_rel > base_rel:
        print(f"\n✅ 句子窗口检索在两个维度上都优于常规分块检索")
    elif sentence_faith > base_faith or sentence_rel > base_rel:
        print(f"\n⚖️  句子窗口检索在某些维度上有优势")
    else:
        print(f"\n❌ 句子窗口检索未显示明显优势")

if __name__ == "__main__":
    asyncio.run(main())