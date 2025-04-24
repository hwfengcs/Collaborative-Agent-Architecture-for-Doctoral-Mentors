from openai import OpenAI
import os
from typing import List, Dict, Any, Optional

class KnowledgeRetrievalModule:
    """
    知识获取模块，负责论文检索和LLM咨询
    """
    def __init__(self, model: str = "deepseek-reasoner"):
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
        # 系统提示词
        system_prompt = """你是一个专业的学术搜索引擎，专注于提供高质量的学术论文信息，特别是互联网内容挖掘、数据分析和实际应用价值方面的研究。"""

        # 用户提示词
        user_prompt = f"""
        请根据以下查询生成{max_results}篇最相关的论文信息，特别关注互联网内容挖掘、数据分析和实际应用价值方面的研究：

        查询: {query}

        请生成与该查询高度相关的{max_results}篇高影响力论文信息，优先选择：
        1. 具有实际应用案例和实验验证的研究
        2. 提出可量化效果的方法论
        3. 解决实际互联网数据挖掘挑战的工作
        4. 在真实大规模数据集上进行验证的研究
        5. 被工业界实际采纳或引用的方法

        每篇论文信息必须包括:
        1. 论文标题
        2. 作者列表及其机构（优先包含学术与工业界合作的研究）
        3. 发表年份(2020-2024之间)
        4. 摘要(200字左右)，突出其实际应用价值和实验结果
        5. 发表期刊/会议(如KDD, WWW, WSDM, SIGIR, CIKM等数据挖掘和信息检索领域顶级会议或期刊)
        6. 引用数和影响力指标（如有）
        7. 主要技术贡献和实际应用场景

        请确保论文内容与查询高度相关，具有明确的实用价值和可实施性。以JSON格式返回结果，格式如下:

        [
          {{
            "title": "论文标题1",
            "authors": "作者1 (机构1), 作者2 (机构2)...",
            "year": "20XX",
            "venue": "会议/期刊名称",
            "citations": "被引用次数（如有）",
            "abstract": "论文摘要（突出实际应用价值和实验结果）...",
            "contributions": "主要技术贡献点",
            "applications": "实际应用场景"
          }},
          {{
            "title": "论文标题2",
            "authors": "作者1 (机构1), 作者2 (机构2)...",
            "year": "20XX",
            "venue": "会议/期刊名称",
            "citations": "被引用次数（如有）",
            "abstract": "论文摘要（突出实际应用价值和实验结果）...",
            "contributions": "主要技术贡献点",
            "applications": "实际应用场景"
          }}
        ]

        只返回JSON数据，不要有其他任何文字。
        """

        try:
            # 确保系统消息在消息序列的开头
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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
        model = "deepseek-reasoner" if use_reasoning else self.model

        # 系统提示词
        system_prompt = """你是一位既有学术背景又有工业界经验的AI研究顾问，专注于LLM-Agent与互联网数据挖掘的交叉领域研究。你提供的建议应基于实证研究和已发表文献，注重实际应用场景和可量化的业务价值。"""

        # 用户提示词
        user_prompt = f"""
        我是一名人工智能专业的博士生，正在研究LLM-Agent与互联网数据挖掘的交叉领域，特别关注实际应用场景和可量化的业务价值。请回答我以下问题:

        {query}

        请在回答中特别注重以下几点：
        1. 提供基于实证研究和已发表文献的具体方法论和技术路线
        2. 引用近期（2020-2024）在实际互联网场景中得到验证的研究成果
        3. 详细说明方法的实现细节、计算复杂度和系统架构
        4. 分析方法在大规模数据集上的性能表现和可扩展性
        5. 提供具体的评估指标和基准测试结果
        6. 讨论方法在实际业务场景中的应用案例和价值
        7. 分析实施过程中可能面临的工程挑战和解决方案

        请提供全面、准确、深入且实用的回答，避免过于理论化或缺乏实证支持的内容。如有可能，请提供代码示例、伪代码或系统架构图的文字描述。
        """

        try:
            # 确保系统消息在消息序列的开头
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3  # 较低的温度以获得更准确的知识
            )

            # 对于推理模型，获取reasoning_content和content
            if use_reasoning:
                try:
                    reasoning = response.choices[0].message.reasoning_content
                    answer = response.choices[0].message.content
                    return {
                        "reasoning": reasoning,
                        "answer": answer
                    }
                except AttributeError:
                    # 如果模型不支持reasoning_content，则只返回content
                    return response.choices[0].message.content
            else:
                return response.choices[0].message.content

        except Exception as e:
            error_message = f"LLM咨询出错: {str(e)}"
            print(error_message)
            return error_message