"""
启动命令：
python ./code/Search2-5-Aliyun-Stream.py
"""

import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import jieba
import os
from typing import List, Dict, Any, Tuple
import requests
from openai import OpenAI

# 存储有效的API密钥（可添加多个）
VALID_API_KEYS = {
    "**********************************": "client1",  # 例如 "abcdefg12345_xyz": "用户A"
    #"你的密钥2": "client2"   # 后续可扩展
}

class DrugSearchSystem:
    """药品数据检索系统"""

    def __init__(self, data_path: str = "data/cleaned_data.json"):
        """初始化药品检索系统

        Args:
            data_path: 药品数据文件路径
        """
        """初始化药品检索系统

            Args:
                data_path: 药品数据文件路径
                index_path: 索引存储目录
            """

        import time  # 新增
        # 记录初始化开始时间
        start_time = time.time()

        # 创建索引目录
        index_path="indexes/"
        os.makedirs(index_path, exist_ok=True)
        self.index_path = index_path
        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            self.drug_data = json.load(f)

        # 初始化向量模型
        # https://huggingface.co/shibing624/text2vec-base-chinese
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

        # 检查索引
        if self._indexes_exist():
            print("检测到已有索引文件，正在加载...")
            load_start = time.time()
            self.load_indexes()
            print(f"索引加载耗时：{time.time() - load_start:.2f}秒")
        else:
            print("未找到索引文件，开始构建索引...")
            build_start = time.time()
            self.build_indexes()
            self.save_indexes()
            print(f"索引构建耗时：{time.time() - build_start:.2f}秒")

        # 打印总耗时
        print(f"系统初始化总耗时：{time.time() - start_time:.2f}秒\n")

        # 定义可查询分类
        self.section_titles = ["处方", "性状", "鉴别", "检查", "含量测定", "类别", "贮藏"]

        # 分类同义词表 - 扩展识别能力
        self.section_synonyms = {
            "处方": ["处方", "配方", "成分", "组成", "组分", "含有", "配比", "配料", "原料", "药材", "构成", "配制",
                     "制剂成分", "组方", "方剂", "处方组成", "药品组成"],
            "性状": ["性状", "外观", "形状", "颜色", "气味", "状态", "外形", "特征", "物理特性", "性质", "质地", "手感",
                     "外貌", "形态", "表观", "色泽", "嗅味", "物态", "外观描述"],
            "鉴别": ["鉴别", "鉴定", "识别", "判断", "辨别", "特征", "特性", "区分", "甄别", "分辨", "检测方法", "确认",
                     "特异性", "区别", "识别方法", "身份确认", "真伪鉴别"],
            "检查": ["检查", "测试", "评估", "验证", "实验", "检测", "质控", "试验", "分析", "监测", "纯度检查",
                     "杂质检查", "品控", "质量标准", "质检", "药检"],
            "含量测定": ["含量测定", "含量", "测定", "浓度", "定量", "检测量", "含量分析", "定量分析", "测量",
                         "含量检测", "定值", "效价测定", "测定法", "量化", "分析测定", "含量要求"],
            "类别": ["类别", "分类", "种类", "药理", "作用", "功效", "类型", "归类", "功能", "适应症", "用途", "疗效",
                     "药效", "药用价值", "适用范围", "治疗作用", "主治", "适用症"],
            "贮藏": ["贮藏", "存储", "保存", "储存条件", "保管", "储存", "包装", "有效期", "保质期", "储藏", "储存方式",
                     "防潮", "密封", "贮存", "保管条件", "避光", "保存条件", "有效日期"]
        }

    def _indexes_exist(self) -> bool:
        """检查索引文件是否存在"""
        required_files = [
            os.path.join(self.index_path, "drug_names.json"),
            os.path.join(self.index_path, "drug_name_vectors.npy"),
            os.path.join(self.index_path, "documents.json"),
            os.path.join(self.index_path, "document_vectors.npy")
        ]
        return all(os.path.exists(f) for f in required_files)

    def save_indexes(self):
        """保存索引到文件"""
        import numpy as np

        # 保存药品名称列表
        with open(os.path.join(self.index_path, "drug_names.json"), "w", encoding="utf-8") as f:
            json.dump(self.drug_names, f, ensure_ascii=False, indent=2)

        # 保存药品名称向量
        np.save(os.path.join(self.index_path, "drug_name_vectors.npy"), self.drug_name_vectors)

        # 保存文档信息
        with open(os.path.join(self.index_path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        # 保存文档向量
        np.save(os.path.join(self.index_path, "document_vectors.npy"), self.document_vectors)

    def load_indexes(self):
        """从文件加载索引"""
        import numpy as np

        # 加载药品名称列表
        with open(os.path.join(self.index_path, "drug_names.json"), "r", encoding="utf-8") as f:
            self.drug_names = json.load(f)

        # 加载药品名称向量
        self.drug_name_vectors = np.load(os.path.join(self.index_path, "drug_name_vectors.npy"))

        # 加载文档信息
        with open(os.path.join(self.index_path, "documents.json"), "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # 加载文档向量
        self.document_vectors = np.load(os.path.join(self.index_path, "document_vectors.npy"))
    
    def build_indexes(self):
        """构建搜索索引"""
        # 药品名称索引
        self.drug_names = [drug["drug_name"] for drug in self.drug_data]
        self.drug_name_vectors = self.model.encode(self.drug_names)

        # 构建分段文档索引
        self.documents = []
        self.document_vectors = []

        for drug in self.drug_data:
            drug_name = drug["drug_name"]

            # 为每个部分创建文档
            for category in drug.get("categories", []):
                section = category.get("section_title", "")
                content = category.get("content", "")

                # 基本文档
                if content:
                    doc_text = f"{drug_name} - {section}: {content}"
                    self.documents.append({
                        "drug_name": drug_name,
                        "section": section,
                        "content": content,
                        "text": doc_text,
                        "type": "basic"
                    })
                    self.document_vectors.append(self.model.encode(doc_text))

                # 测试部分文档
                if "tests" in category and category["tests"]:
                    for test in category["tests"]:
                        test_name = test.get("test_name", "")
                        procedure = test.get("procedure", "")
                        sampling = test.get("sampling", "")

                        if procedure:
                            test_doc = f"{drug_name} - {section} - {test_name}: {procedure}"
                            self.documents.append({
                                "drug_name": drug_name,
                                "section": section,
                                "test_name": test_name,
                                "sampling": sampling,
                                "procedure": procedure,
                                "text": test_doc,
                                "type": "test"
                            })
                            self.document_vectors.append(self.model.encode(test_doc))

        # 转换为numpy数组以提高检索效率
        self.document_vectors = np.array(self.document_vectors)

    def identify_query_type(self, query: str) -> List[str]:
        """识别查询分类

        Args:
            query: 用户问题

        Returns:
            查询相关的分类列表
        """
        words = jieba.lcut(query)
        identified_sections = set()

        # 检查查询中的关键词是否匹配任何分类的同义词
        for section, synonyms in self.section_synonyms.items():
            for word in words:
                if word in synonyms:
                    identified_sections.add(section)

        # 如果没有识别到分类，尝试向量相似度匹配
        if not identified_sections:
            query_vec = self.model.encode(query)
            section_vecs = self.model.encode(self.section_titles)
            similarities = cosine_similarity([query_vec], section_vecs)[0]

            # 打印每个查询类别的相似度判断
            print("-" * 50)
            print(f"查询类别相似度判断：")
            for i, section in enumerate(self.section_titles):
                print(f"{section}: {similarities[i]:.4f}")
            print("-" * 50)

            max_sim_idx = np.argmax(similarities)
            if similarities[max_sim_idx] > 0.5:  # 设置相似度阈值
                identified_sections.add(self.section_titles[max_sim_idx])

        return list(identified_sections) if identified_sections else ["全部"]

    def extract_drug_names(self, query: str) -> List[str]:
        """从查询中提取药品名称

        Args:
            query: 用户问题

        Returns:
            查询中提到的药品名称列表
        """
        mentioned_drugs = []

        # 打印所有可能的药品名称供调试
        print(f"可能的药品名称列表: {', '.join(self.drug_names[:5])}... 等{len(self.drug_names)}个")

        # 精确匹配
        for drug_name in self.drug_names:
            if drug_name in query:
                mentioned_drugs.append(drug_name)
                print(f"精确匹配到药品: {drug_name}")

        # 如果没有找到精确匹配，尝试模糊匹配
        if not mentioned_drugs:
            query_vec = self.model.encode(query)
            similarities = cosine_similarity([query_vec], self.drug_name_vectors)[0]

            # 打印相似度最高的几个药品名称
            top_indices = np.argsort(similarities)[::-1][:5]  # 获取前5个最相似的
            print("相似度最高的几个药品:")
            for idx in top_indices:
                print(f"  {self.drug_names[idx]}: {similarities[idx]:.4f}")

            # 获取最相似的药品名称(相似度阈值提高到0.6)
            top_indices = np.argsort(similarities)[::-1][:3]  # 获取前3个最相似的
            for idx in top_indices:
                if similarities[idx] > 0.5:  # 设置相似度阈值
                    mentioned_drugs.append(self.drug_names[idx])
                    print(f"模糊匹配到药品: {self.drug_names[idx]} (相似度: {similarities[idx]:.4f})")

        print(f"最终识别到的药品: {mentioned_drugs}")
        return mentioned_drugs

    def semantic_search(self, query: str, top_k: int = 5,
                        sections: List[str] = None,
                        drug_names: List[str] = None) -> List[Dict]:
        """语义搜索

        Args:
            query: 用户问题
            top_k: 返回的结果数量
            sections: 限制搜索的分类
            drug_names: 限制搜索的药品名称

        Returns:
            搜索结果列表
        """
        # 编码查询
        query_vec = self.model.encode(query)

        # 计算相似度
        similarities = cosine_similarity([query_vec], self.document_vectors)[0]

        # 应用过滤器
        filtered_indices = np.arange(len(self.documents))

        if sections and "全部" not in sections:
            section_mask = np.array([doc["section"] in sections for doc in self.documents])
            filtered_indices = filtered_indices[section_mask]

        if drug_names:
            # 使用已过滤的索引创建drug_mask
            drug_mask = np.array([self.documents[idx]["drug_name"] in drug_names for idx in filtered_indices])
            filtered_indices = filtered_indices[drug_mask]

        # 如果过滤后没有结果，返回空列表
        if len(filtered_indices) == 0:
            return []

        # 对过滤后的文档排序
        filtered_similarities = similarities[filtered_indices]
        ranked_indices = filtered_indices[np.argsort(filtered_similarities)[::-1][:top_k]]

        # 返回结果
        results = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append({
                "document": doc,
                "similarity": float(similarities[idx]),
                "score": float(similarities[idx])
            })

        return results

    def process_query(self, query: str) -> Dict[str, Any]:
        """处理用户查询

        Args:
            query: 用户问题

        Returns:
            处理结果，包含识别的类型、药品名称和检索结果
        """
        # 识别查询类型
        sections = self.identify_query_type(query)

        # 提取药品名称
        drug_names = self.extract_drug_names(query)

        # 对于多药品查询，为每个药品关联查询类别
        if len(drug_names) > 1:
            # 验证识别到的药品确实存在于数据库中
            valid_drug_names = []
            drug_sections = {}  # 存储每个药品对应的查询类别

            for drug_name in drug_names:
                for drug in self.drug_data:
                    if drug["drug_name"] == drug_name:
                        valid_drug_names.append(drug_name)
                        # 默认使用识别到的全局类别，没有则使用"全部"
                        drug_sections[drug_name] = sections if "全部" not in sections else ["全部"]
                        break

            # 更新为有效的药品名称
            drug_names = valid_drug_names
            print(f"有效药品名称: {drug_names}")

            # 执行多药品的语义搜索 - 为每个药品单独搜索
            search_results = []
            for drug_name in drug_names:
                drug_results = self.semantic_search(
                    query=query,
                    top_k=3,  # 每个药品返回较少结果，避免总结果过多
                    sections=drug_sections.get(drug_name, ["全部"]) if "全部" not in drug_sections.get(drug_name, [
                        "全部"]) else None,
                    drug_names=[drug_name]
                )
                search_results.extend(drug_results)

            print(f"is_multi_drug: 1")
            return {
                "query": query,
                "identified_sections": sections,
                "identified_drugs": drug_names,
                "drug_sections": drug_sections,  # 新增每个药品的查询类别
                "search_results": search_results,
                "is_multi_drug": True  # 标记为多药品查询
            }
        else:
            # 单药品查询处理逻辑保持不变
            search_results = self.semantic_search(
                query=query,
                top_k=5,
                sections=sections if "全部" not in sections else None,
                drug_names=drug_names if drug_names else None
            )

            return {
                "query": query,
                "identified_sections": sections,
                "identified_drugs": drug_names,
                "search_results": search_results,
                "is_multi_drug": False
            }

    def generate_answer(self, query_results: Dict[str, Any], api_key) -> str:
        """调用LLM-API生成回答

        Args:
            query_results: 查询处理结果
            api_key: API密钥

        Returns:
            生成的回答
        """
        # 构建上下文
        context = ""

        # 添加识别的药品名称和分类信息
        if query_results["identified_drugs"]:
            if len(query_results["identified_drugs"]) == 1:
                context += f"识别到的药品: {query_results['identified_drugs'][0]}\n\n"
            else:
                context += f"识别到的多个药品: {', '.join(query_results['identified_drugs'])}\n\n"

        # 多药品查询处理
        print(f"is_multi_drug: {query_results.get('is_multi_drug')}")
        if query_results.get("is_multi_drug") and query_results["identified_drugs"]:
            drug_sections = query_results.get("drug_sections", {})

            # 显示每个药品及其对应的查询类别
            if drug_sections:
                context += "查询的药品及分类:\n"
                for drug_name, sections in drug_sections.items():
                    context += f"- {drug_name}: {', '.join(sections)}\n"
                context += "\n"

            # 跟踪找到了哪些药品的信息
            found_drugs = []

            for drug_name in query_results["identified_drugs"]:
                found_drug = False
                context += f"【{drug_name}】的详细信息:\n"

                # 获取该药品的查询类别
                drug_specific_sections = drug_sections.get(drug_name, ["全部"])

                # 从药品数据中查找匹配的药品
                for drug in self.drug_data:
                    if drug["drug_name"] == drug_name:
                        found_drug = True
                        found_drugs.append(drug_name)
                        print(f"找到药品 '{drug_name}' 的数据")

                        # 根据该药品的查询类别添加信息
                        for category in drug.get("categories", []):
                            section = category.get("section_title", "")
                            content = category.get("content", "")

                            # 如果是全部类别或者该分类在查询类别中
                            if "全部" in drug_specific_sections or section in drug_specific_sections:
                                if content:
                                    context += f"* {section}: {content}\n"
                                    print(f"  - 添加 {section} 信息")

                                # 添加测试部分的信息
                                if "tests" in category and category["tests"]:
                                    for test in category["tests"]:
                                        test_name = test.get("test_name", "")
                                        procedure = test.get("procedure", "")

                                        if procedure:
                                            context += f"  - {test_name}: {procedure}\n"
                                            print(f"    - 添加测试 {test_name}")

                        context += "\n"
                        break

                if not found_drug:
                    context += f"* 未找到 '{drug_name}' 的详细信息\n\n"
                    print(f"警告: 未找到药品 '{drug_name}' 的数据")

            # 检查是否所有药品都找到了
            print(f"识别到的药品: {query_results['identified_drugs']}")
            print(f"找到数据的药品: {found_drugs}")

            missing_drugs = [drug for drug in query_results["identified_drugs"] if drug not in found_drugs]
            if missing_drugs:
                context += f"\n注意: 未能找到以下药品的完整信息: {', '.join(missing_drugs)}\n\n"

        # 单药品查询或一般查询处理
        elif "全部" in query_results["identified_sections"] and query_results["identified_drugs"]:
            context += f"查询的分类: 全部药品信息\n\n"

            # 跟踪找到了哪些药品的信息
            found_drugs = []

            for drug_name in query_results["identified_drugs"]:
                found_drug = False
                context += f"【{drug_name}】的详细信息:\n"

                # 从药品数据中查找匹配的药品
                for drug in self.drug_data:
                    if drug["drug_name"] == drug_name:
                        found_drug = True
                        found_drugs.append(drug_name)
                        print(f"找到药品 '{drug_name}' 的数据")

                        # 添加所有分类的信息
                        for category in drug.get("categories", []):
                            section = category.get("section_title", "")
                            content = category.get("content", "")

                            if content:
                                context += f"* {section}: {content}\n"
                                print(f"  - 添加 {section} 信息")

                            # 添加测试部分的信息
                            if "tests" in category and category["tests"]:
                                for test in category["tests"]:
                                    test_name = test.get("test_name", "")
                                    procedure = test.get("procedure", "")

                                    if procedure:
                                        context += f"  - {test_name}: {procedure}\n"
                                        print(f"    - 添加测试 {test_name}")

                        context += "\n"
                        break

                if not found_drug:
                    context += f"* 未找到 '{drug_name}' 的详细信息\n\n"
                    print(f"警告: 未找到药品 '{drug_name}' 的数据")

            # 检查是否所有药品都找到了
            print(f"识别到的药品: {query_results['identified_drugs']}")
            print(f"找到数据的药品: {found_drugs}")

            missing_drugs = [drug for drug in query_results["identified_drugs"] if drug not in found_drugs]
            if missing_drugs:
                context += f"\n注意: 未能找到以下药品的完整信息: {', '.join(missing_drugs)}\n\n"
        # 普通查询处理逻辑
        else:
            if "全部" not in query_results["identified_sections"]:
                context += f"查询的分类: {', '.join(query_results['identified_sections'])}\n\n"

            # 添加搜索结果作为上下文
            context += "相关信息:\n"
            for i, result in enumerate(query_results["search_results"], 1):
                doc = result["document"]
                context += f"{i}. {doc['text']}\n\n"

        # 如果没有API密钥，返回上下文信息
        if not api_key:
            return f"查询: {query_results['query']}\n\n{context}\n(未提供API密钥，无法生成完整回答)"

        # 调用API生成回答
        return self._call_llm_api(query_results["query"], context, api_key)

    def _call_llm_api(self, query: str, context: str, api_key: str) -> str:
        """调用阿里云大语言模型API（流式输出）

        Args:
            query: 用户问题
            context: 上下文信息
            api_key: API密钥

        Returns:
            生成的回答
        """
        # 构建提示信息
        system_message = """
        角色：你是药检所的专业的药品信息顾问，擅长回答药检所工作人员各种关于药品的问题。
        任务：模拟人与人自然的对话，根据程序检索到的药典信息给出精确回答，不要删减，不要修改，作为药检所的专业的药品信息顾问，回答要像人与人的自然对话。
        禁忌：不要删减，不要修改，不要复杂的句式，不要多余的符号，不要用markdown格式。
        要求：要原封不动地采用关键信息，不要对关键信息进行修改（数字、通则0621、光谱集11图等等）。
        要求：对于程序检索到的重复的表头信息可以删除。
        要求：如果信息不足，请说明无法回答的原因。
        语言：回答时，请使用中文。
        """

        user_message = f"""
        药检所工作人员的问题: {query}\n
        程序检索到的药典信息:\n{context}
        药检所工作人员的问题: {query}
        """
        LLM_Model = "deepseek-v3"

        # 定义标点符号列表，用于分割句子
        PUNCTUATIONS = ["：", ":", "。", "！", "？", "；", "!", "?", ";", "\n"]

        try:
            # 使用OpenAI客户端调用阿里云API
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            print(f"调用阿里云API，使用模型：{LLM_Model}（流式输出）")

            # 创建流式聊天完成请求
            completion = client.chat.completions.create(
                model=LLM_Model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=0.7,
                max_tokens=5000,
                stream=True  # 启用流式输出
            )

            print("\n" + "-" * 50)
            print("回答开始流式输出：")
            print("-" * 50)

            full_content = ""  # 保存完整内容
            current_sentence = ""  # 当前正在构建的句子
            sentence_count = 1  # 句子计数

            # 处理流式输出
            for chunk in completion:
                if chunk.choices:
                    # 获取当前片段的内容
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        full_content += content
                        current_sentence += content

                        # 检查是否包含标点符号
                        for punct in PUNCTUATIONS:
                            if punct in content:
                                # 找到标点符号的位置
                                parts = current_sentence.split(punct)
                                if len(parts) > 1:
                                    # 提取完整的句子
                                    for i in range(len(parts) - 1):
                                        complete_sentence = parts[i] + punct
                                        print(f"句子{sentence_count}：{complete_sentence}")
                                        sentence_count += 1

                                    # 更新当前句子为剩余部分
                                    current_sentence = parts[-1]

            # 如果最后还有未打印的句子
            if current_sentence.strip():
                print(f"句子{sentence_count}：{current_sentence}")

            print("-" * 50)
            print("流式输出完成")
            print("-" * 50)

            return full_content

        except Exception as e:
            # 处理错误
            error_message = f"API请求错误: {str(e)}"
            print(error_message)
            return f"{error_message}\n\n这是找到的相关信息:\n{context}"


def main():
    """主函数，用于测试"""
    # 初始化系统
    search_system = DrugSearchSystem()

    # 替换为你的阿里云API密钥
    # 获取API密钥方法请参考：https://help.aliyun.com/zh/model-studio/getting-started/connect-to-dashscope
    api_key = "********************************"  # 此为示例密钥，请替换为你的实际密钥

    # 处理用户查询的例子
    while True:
        query = input("\n请输入您的问题 (输入'exit'退出): ")
        if query.lower() == 'exit':
            break

        # 处理查询
        results = search_system.process_query(query)

        # 输出查询识别结果
        print("\n" + "-" * 50)
        print(f"查询: {query}")
        print(f"识别的分类: {', '.join(results['identified_sections'])}")
        print(f"识别的药品: {', '.join(results['identified_drugs']) if results['identified_drugs'] else '无'}")
        print("-" * 50)

        # 生成回答(已在流式输出中显示)
        search_system.generate_answer(results, api_key)


if __name__ == "__main__":
    main()

