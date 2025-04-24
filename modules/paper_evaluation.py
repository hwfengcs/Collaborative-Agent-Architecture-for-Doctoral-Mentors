from openai import OpenAI
import os
from typing import Dict, Any, List, Optional

class PaperEvaluationModule:
    """
    论文评估模块，负责评估论文质量并提供修改建议
    """
    def __init__(self, model: str = "deepseek-reasoner"):
        """
        初始化论文评估模块

        Args:
            model: 使用的大模型名称
        """
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        # 评估维度及其权重
        self.evaluation_dimensions = {
            "创新性": 0.25,            # 理论或方法的原创贡献
            "学术严谨性": 0.25,         # 研究方法、实验设计的科学性
            "实用价值": 0.20,          # 在抖音等平台的应用潜力
            "写作质量": 0.15,          # 结构清晰度、论证逻辑性
            "实验评估": 0.15           # 实验设计和结果分析
        }

    def evaluate_paper(self, paper_content: str, target_venue: str = None) -> Dict[str, Any]:
        """
        评估论文质量

        Args:
            paper_content: 论文内容
            target_venue: 目标发表会议/期刊

        Returns:
            评估结果，包含总体评分、各维度评分和修改建议
        """
        prompt = f"""
        请对以下论文进行全面评估：

        {"目标发表会议/期刊：" + target_venue if target_venue else ""}

        论文内容：
        {paper_content}

        请从以下维度进行评估，并给出1-10分的评分（10分为最高）：

        1. 创新性：理论或方法的原创贡献
        2. 学术严谨性：研究方法、实验设计的科学性
        3. 实用价值：在实际应用场景（如抖音平台）的应用潜力
        4. 写作质量：结构清晰度、论证逻辑性
        5. 实验评估：实验设计和结果分析的质量

        对于每个维度，请给出详细的评价和具体的改进建议。

        最后，请给出总体评分（1-10分）和总体修改建议。

        评估结果请以JSON格式返回，格式如下：

        {{
          "总体评分": 分数,
          "维度评分": {{
            "创新性": {{
              "分数": 分数,
              "评价": "评价内容",
              "改进建议": ["建议1", "建议2", ...]
            }},
            "学术严谨性": {{
              "分数": 分数,
              "评价": "评价内容",
              "改进建议": ["建议1", "建议2", ...]
            }},
            // 其他维度...
          }},
          "总体修改建议": ["建议1", "建议2", ...]
        }}

        只返回JSON数据，不要有其他任何文字。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # 较低的温度以获得更一致的评估
            )
            result = response.choices[0].message.content

            # 理想情况下应该解析JSON，但为简化处理，直接返回文本
            return result

        except Exception as e:
            error_message = f"论文评估出错: {str(e)}"
            print(error_message)
            return error_message

    def generate_improvement_plan(self, paper_content: str, evaluation_result: str) -> str:
        """
        基于评估结果生成改进计划

        Args:
            paper_content: 论文内容
            evaluation_result: 评估结果

        Returns:
            详细的改进计划
        """
        prompt = f"""
        您是一位经验丰富的学术导师，请基于以下论文内容和评估结果，生成一份详细的改进计划：

        评估结果：
        {evaluation_result}

        请提供一份详细的改进计划，包括：

        1. 优先级最高的3个改进方向
        2. 针对每个方向的具体行动步骤
        3. 如何保持论文的现有优势
        4. 可以参考的优秀论文或研究方法
        5. 时间规划建议

        请确保改进计划是具体、可操作的，而不是泛泛而谈。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content

        except Exception as e:
            error_message = f"生成改进计划出错: {str(e)}"
            print(error_message)
            return error_message

    def check_publication_readiness(self, paper_content: str, target_venue: str) -> Dict[str, Any]:
        """
        检查论文是否达到发表标准

        Args:
            paper_content: 论文内容
            target_venue: 目标发表会议/期刊

        Returns:
            论文发表就绪性评估
        """
        prompt = f"""
        请作为学术审稿人，评估以下论文是否达到在{target_venue}发表的标准：

        论文内容：
        {paper_content}

        请回答以下问题：

        1. 这篇论文达到{target_venue}发表的标准了吗？（是/否）
        2. 如果没有达到，还有哪些关键问题需要解决？
        3. 如果已经达到，有哪些可能进一步提升其被接收机会的改进点？
        4. 与该领域近期在{target_venue}发表的论文相比，这篇论文的优势和劣势是什么？
        5. 预计该论文在提交后可能收到的主要批评是什么？如何提前应对？

        请提供具体、建设性的反馈，以JSON格式返回结果：

        {{
          "达到发表标准": true/false,
          "关键问题": ["问题1", "问题2", ...],
          "改进建议": ["建议1", "建议2", ...],
          "优势": ["优势1", "优势2", ...],
          "劣势": ["劣势1", "劣势2", ...],
          "可能的批评": ["批评1", "批评2", ...],
          "应对策略": ["策略1", "策略2", ...]
        }}

        只返回JSON数据，不要有其他任何文字。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result = response.choices[0].message.content

            # 直接返回文本结果
            return result

        except Exception as e:
            error_message = f"发表就绪性检查出错: {str(e)}"
            print(error_message)
            return error_message