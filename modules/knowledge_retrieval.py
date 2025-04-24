from openai import OpenAI
import os
from typing import List, Dict, Any, Optional

class KnowledgeRetrievalModule:
    """
    知识获取模块，负责论文检索和LLM咨询
    """
    def __init__(self, model: str = "deepseek-r1"):
        """
        初始化知识获取模块

        Args:
            model: 使用的大模型名称
        """
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        模拟论文检索功能，通过LLM生成假设的论文信息

        在实际应用中，这里可以连接到实际的学术API，如Semantic Scholar、Google Scholar等

        Args:
            query: 检索关键词
            max_results: 最大返回结果数

        Returns:
            论文信息列表 [{"title": "论文标题", "authors": "作者", "year": "年份", "abstract": "摘要", "link": "链接"}]
        """
        prompt = f"""
        作为一个学术搜索引擎，请根据以下查询生成{max_results}篇最相关的论文信息：

        查询: {query}

        请生成可能与该查询相关的{max_results}篇高水平、最新的论文信息，包括:
        1. 论文标题
        2. 作者列表
        3. 发表年份(2020-2024之间)
        4. 摘要(200字左右)
        5. 发表期刊/会议(如CVPR, NeurIPS, ICLR, ACL等顶级会议或期刊)

        请确保论文内容与查询相关，且具有学术价值和创新性。以JSON格式返回结果，格式如下:

        [
          {{
            "title": "论文标题1",
            "authors": "作者1, 作者2...",
            "year": "20XX",
            "venue": "会议/期刊名称",
            "abstract": "论文摘要...",
          }},
          {{
            "title": "论文标题2",
            "authors": "作者1, 作者2...",
            "year": "20XX",
            "venue": "会议/期刊名称",
            "abstract": "论文摘要...",
          }}
        ]

        只返回JSON数据，不要有其他任何文字。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            result = response.choices[0].message.content

            # 这里可以添加JSON解析逻辑，但为简化，直接返回文本结果
            return result

        except Exception as e:
            error_message = f"论文检索出错: {str(e)}"
            print(error_message)
            return error_message

    def consult_llm(self, query: str, use_reasoning: bool = False) -> str:
        """
        向LLM咨询知识

        Args:
            query: 咨询问题
            use_reasoning: 是否使用推理模型

        Returns:
            LLM的回答
        """
        model = "deepseek-r1" if use_reasoning else self.model

        prompt = f"""
        我是一名人工智能专业的博士生，正在研究LLM-Agent与数据挖掘的交叉领域。请作为专业的AI研究顾问，回答我以下问题:

        {query}

        请提供全面、准确、深入的回答，如果有相关的研究或方法论，请一并介绍。
        """

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # 较低的温度以获得更准确的知识
            )

            # 由于deepseek-r1模型可能不支持reasoning_content属性，统一返回content
            return response.choices[0].message.content

        except Exception as e:
            error_message = f"LLM咨询出错: {str(e)}"
            print(error_message)
            return error_message