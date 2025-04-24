from agents.phd_student import PhDStudentAgent
from agents.academic_advisor import AcademicAdvisorAgent
from agents.industry_advisor import IndustryAdvisorAgent
from modules.knowledge_retrieval import KnowledgeRetrievalModule
from modules.paper_evaluation import PaperEvaluationModule
import json
import os
from typing import Dict, Any, List, Optional

class AgentCoordinator:
    """
    Agent协调器，负责管理Agent之间的交互和系统工作流程
    """
    def __init__(
        self,
        phd_student: PhDStudentAgent,
        academic_advisor: AcademicAdvisorAgent,
        industry_advisor: IndustryAdvisorAgent,
        knowledge_module: KnowledgeRetrievalModule,
        evaluation_module: PaperEvaluationModule
    ):
        """
        初始化协调器

        Args:
            phd_student: 博士生Agent
            academic_advisor: 高校导师Agent
            industry_advisor: 企业导师Agent
            knowledge_module: 知识获取模块
            evaluation_module: 论文评估模块
        """
        self.phd_student = phd_student
        self.academic_advisor = academic_advisor
        self.industry_advisor = industry_advisor
        self.knowledge_module = knowledge_module
        self.evaluation_module = evaluation_module

        # 记录系统状态
        self.current_phase = "initialization"  # 当前阶段
        self.interaction_history = []  # 交互历史
        self.paper_drafts = []  # 论文草稿历史

    def add_to_history(self, speaker: str, message: str) -> None:
        """
        添加交互记录到历史

        Args:
            speaker: 发言者
            message: 消息内容
        """
        self.interaction_history.append({
            "speaker": speaker,
            "message": message,
            "phase": self.current_phase
        })
        print(f"\n[{speaker}]: {message}\n")

    def start_interaction(self) -> None:
        """
        启动系统交互
        """
        print("\n===== 多Agent博士生指导系统启动 =====\n")

        # 初始化阶段
        self.current_phase = "initialization"
        self._initialization_phase()

        # 研究执行阶段
        self.current_phase = "research_execution"
        self._research_execution_phase()

        # 论文撰写阶段
        self.current_phase = "paper_writing"
        self._paper_writing_phase()

        # 论文优化阶段
        self.current_phase = "paper_optimization"
        self._paper_optimization_phase()

        # 定稿发表阶段
        self.current_phase = "paper_finalization"
        self._paper_finalization_phase()

        print("\n===== 系统交互完成 =====\n")

    def _initialization_phase(self) -> None:
        """
        初始化阶段的交互流程
        """
        print("\n----- 初始化阶段开始 -----\n")

        # 博士生提出初步研究方向
        initial_topic = self.phd_student.get_response(
            "作为一名人工智能专业的博士生，我需要确定一个研究方向。请思考一个在LLM-Agent与抖音数据挖掘交叉领域的有价值研究方向。"
        )
        self.add_to_history("博士生", initial_topic)

        # 高校导师提供学术建议
        academic_feedback = self.academic_advisor.answer_question(initial_topic)
        self.add_to_history("高校导师", academic_feedback)

        # 企业导师提供产业视角建议
        industry_feedback = self.industry_advisor.answer_question(initial_topic)
        self.add_to_history("企业导师", industry_feedback)

        # 博士生整合反馈，确定研究方向
        direction_prompt = f"""
        基于两位导师的反馈：

        高校导师的建议：{academic_feedback}

        企业导师的建议：{industry_feedback}

        请整合两位导师的意见，确定一个明确的研究方向和初步的研究计划。研究方向应该在LLM-Agent与抖音数据挖掘的交叉领域，既有学术价值又有产业应用前景。
        """

        final_direction = self.phd_student.get_response(direction_prompt)
        self.add_to_history("博士生", final_direction)

        # 保存研究方向
        self.phd_student.set_research_topic(final_direction)

        # 博士生制定研究计划
        research_plan_prompt = """
        基于确定的研究方向，请制定一个详细的研究计划，包括：

        1. 研究问题和目标
        2. 文献调研计划
        3. 方法论和技术路线
        4. 实验设计
        5. 预期成果
        6. 时间规划

        请尽可能详细地描述每个部分。
        """

        research_plan = self.phd_student.get_response(research_plan_prompt)
        self.add_to_history("博士生", f"我的研究计划如下：\n\n{research_plan}")

        # 保存研究计划
        self.phd_student.set_research_plan(research_plan)

        # 高校导师审核研究计划
        academic_review = self.academic_advisor.review_research_plan(research_plan)
        self.add_to_history("高校导师", academic_review)

        # 企业导师审核研究计划
        industry_review = self.industry_advisor.review_research_plan(research_plan)
        self.add_to_history("企业导师", industry_review)

        # 博士生根据反馈调整研究计划
        plan_revision_prompt = f"""
        基于两位导师对研究计划的评价：

        高校导师的评价：{academic_review}

        企业导师的评价：{industry_review}

        请修改你的研究计划，回应导师的建议和关注点。
        """

        revised_plan = self.phd_student.get_response(plan_revision_prompt)
        self.add_to_history("博士生", f"我修改后的研究计划：\n\n{revised_plan}")

        # 更新研究计划
        self.phd_student.set_research_plan(revised_plan)

        # 阶段总结 - 为每个代理创建阶段记忆
        phase_context = f"""
        研究方向: {self.phd_student.research_topic}
        研究计划: {self.phd_student.research_plan}
        """

        # 博士生总结
        phd_summary = self.phd_student.summarize_phase("initialization", phase_context)
        self.add_to_history("系统", f"[博士生阶段总结] {phd_summary}")

        # 高校导师总结
        academic_summary = self.academic_advisor.summarize_phase("initialization", phase_context)
        self.add_to_history("系统", f"[高校导师阶段总结] {academic_summary}")

        # 企业导师总结
        industry_summary = self.industry_advisor.summarize_phase("initialization", phase_context)
        self.add_to_history("系统", f"[企业导师阶段总结] {industry_summary}")

        print("\n----- 初始化阶段完成 -----\n")

    def _research_execution_phase(self) -> None:
        """
        研究执行阶段的交互流程
        """
        print("\n----- 研究执行阶段开始 -----\n")

        # 博士生进行文献调研
        literature_query = f"基于我的研究方向: {self.phd_student.research_topic}，请推荐相关的高水平论文"

        self.add_to_history("博士生", f"我需要检索与我研究方向相关的文献: {literature_query}")

        literature_results = self.knowledge_module.search_papers(literature_query)
        self.add_to_history("知识获取模块", f"文献检索结果:\n\n{literature_results}")

        # 博士生咨询高校导师关于文献的问题
        literature_question = self.phd_student.ask_question(
            "academic",
            f"基于检索到的文献，我想请教一下这些论文中的方法论和理论框架是否适合我的研究方向？有哪些值得借鉴的点？"
        )
        self.add_to_history("博士生", literature_question)

        academic_literature_advice = self.academic_advisor.answer_question(literature_question)
        self.add_to_history("高校导师", academic_literature_advice)

        # 博士生咨询企业导师关于实际应用的问题
        application_question = self.phd_student.ask_question(
            "industry",
            f"在抖音的实际业务场景中，这些论文中的方法是否有实际应用价值？存在哪些实际落地的挑战？"
        )
        self.add_to_history("博士生", application_question)

        industry_application_advice = self.industry_advisor.answer_question(application_question)
        self.add_to_history("企业导师", industry_application_advice)

        # 博士生利用LLM获取更多专业知识
        knowledge_query = "在LLM-Agent与数据挖掘结合的场景中，有哪些关键技术挑战和最新解决方案？"
        self.add_to_history("博士生", f"我想咨询一下专业知识: {knowledge_query}")

        knowledge_response = self.knowledge_module.consult_llm(knowledge_query, use_reasoning=True)
        if isinstance(knowledge_response, dict):
            # 对于推理模型的输出
            self.add_to_history("知识获取模块", f"思考过程: {knowledge_response['reasoning']}\n\n回答: {knowledge_response['answer']}")
        else:
            # 对于普通输出
            self.add_to_history("知识获取模块", f"回答: {knowledge_response}")

        # 博士生提出研究方法问题
        method_question = self.phd_student.ask_question(
            "academic",
            "基于我的研究方向和文献调研，我计划采用以下研究方法。这种方法是否合适？有哪些需要注意的地方？"
        )
        self.add_to_history("博士生", method_question)

        academic_method_advice = self.academic_advisor.answer_question(method_question)
        self.add_to_history("高校导师", academic_method_advice)

        # 博士生提出技术实现问题
        implementation_question = self.phd_student.ask_question(
            "industry",
            "对于我的研究方向，在实际工程实现时可能面临哪些技术挑战？有什么建议可以帮助我克服这些挑战？"
        )
        self.add_to_history("博士生", implementation_question)

        industry_implementation_advice = self.industry_advisor.answer_question(implementation_question)
        self.add_to_history("企业导师", industry_implementation_advice)

        # 博士生汇报研究进展
        progress_report_prompt = f"""
        请撰写一份研究进展报告，总结到目前为止的以下内容：

        1. 研究方向和问题的明确化
        2. 文献调研的主要发现
        3. 确定的研究方法和技术路线
        4. 已解决和尚未解决的技术挑战
        5. 下一步计划

        请基于之前的交流内容来撰写这份报告。
        """

        progress_report = self.phd_student.get_response(progress_report_prompt)
        self.add_to_history("博士生", f"研究进展报告：\n\n{progress_report}")

        # 阶段总结 - 为每个代理创建阶段记忆
        phase_context = f"""
        研究方向: {self.phd_student.research_topic}
        研究计划: {self.phd_student.research_plan}
        研究进展报告: {progress_report}
        """

        # 在进入下一阶段前，注入之前阶段的记忆
        self.phd_student.inject_memories_to_context(["initialization"])
        self.academic_advisor.inject_memories_to_context(["initialization"])
        self.industry_advisor.inject_memories_to_context(["initialization"])

        # 博士生总结
        phd_summary = self.phd_student.summarize_phase("research_execution", phase_context)
        self.add_to_history("系统", f"[博士生阶段总结] {phd_summary}")

        # 高校导师总结
        academic_summary = self.academic_advisor.summarize_phase("research_execution", phase_context)
        self.add_to_history("系统", f"[高校导师阶段总结] {academic_summary}")

        # 企业导师总结
        industry_summary = self.industry_advisor.summarize_phase("research_execution", phase_context)
        self.add_to_history("系统", f"[企业导师阶段总结] {industry_summary}")

        print("\n----- 研究执行阶段完成 -----\n")

    def _paper_writing_phase(self) -> None:
        """
        论文撰写阶段的交互流程
        """
        print("\n----- 论文撰写阶段开始 -----\n")

        # 博士生请教论文结构和理论框架
        structure_question = self.phd_student.ask_question(
            "academic",
            "我准备开始撰写论文，请您指导：1）对于我们的研究方向，应采用什么样的理论框架最能体现学术贡献？2）论文应如何组织结构才能严谨地展示研究成果？3）有哪些顶级期刊的论文结构和理论框架值得我参考？"
        )
        self.add_to_history("博士生", structure_question)

        academic_structure_advice = self.academic_advisor.answer_question(structure_question)
        self.add_to_history("高校导师", academic_structure_advice)

        # 博士生请教实验设计和评估方法
        experiment_question = self.phd_student.ask_question(
            "industry",
            "对于我们的研究，我需要设计严谨的实验来验证方法的有效性。请您指导：1）应该设计哪些对照实验和消融实验？2）如何确保实验的统计显著性和可复现性？3）在抖音这样的实际场景中，如何设计全面的评估指标体系？4）如何有效展示实验结果以支持我们的理论主张？"
        )
        self.add_to_history("博士生", experiment_question)

        industry_experiment_advice = self.industry_advisor.answer_question(experiment_question)
        self.add_to_history("企业导师", industry_experiment_advice)

        # 博士生请教文献综述方法
        literature_question = self.phd_student.ask_question(
            "academic",
            "为了确保我的相关工作部分全面且深入，请您指导：1）我应该如何系统地组织和分析现有文献？2）如何有效地识别和突出现有方法的局限性？3）如何将我的工作与现有研究明确区分？4）有哪些最新的研究趋势和方法是我必须涵盖的？"
        )
        self.add_to_history("博士生", literature_question)

        academic_literature_advice = self.academic_advisor.answer_question(literature_question)
        self.add_to_history("高校导师", academic_literature_advice)

        # 博士生撰写论文初稿（摘要和引言）
        paper_draft_prompt = f"""
        请基于我们的研究方向、文献调研结果以及导师的建议，撰写一篇高质量学术论文的摘要和引言部分。

        研究方向：{self.phd_student.research_topic}

        理论框架与结构参考：{academic_structure_advice}

        实验设计参考：{industry_experiment_advice}

        文献综述方法：{academic_literature_advice}

        请撰写：
        1. 摘要（250-300字）：应包含研究背景、问题定义、方法创新点、主要实验结果和理论贡献
        2. 引言（至少1500字）：必须包含以下要素：
           - 研究背景与领域现状（至少300字）
           - 现有方法的系统性分析与局限性（至少500字）
           - 明确的研究问题定义与挑战（至少200字）
           - 本文方法的理论基础与创新点（至少300字）
           - 主要贡献的详细阐述（至少200字）
           - 论文结构概述

        请确保内容：
        - 学术语言严谨，避免模糊或夸大的表述
        - 每个观点都有理论依据或实证支持
        - 清晰界定研究范围和边界条件
        - 准确引用相关文献（至少10篇）
        - 逻辑结构严密，论证过程完整
        """

        initial_draft = self.phd_student.get_response(paper_draft_prompt)
        self.add_to_history("博士生", f"我的论文初稿（摘要和引言部分）：\n\n{initial_draft}")

        # 保存初稿
        self.phd_student.update_paper_draft(initial_draft)
        self.paper_drafts.append({
            "version": 1,
            "content": initial_draft,
            "phase": self.current_phase
        })

        # 高校导师评价初稿
        academic_draft_review = self.academic_advisor.review_paper_draft(initial_draft)
        self.add_to_history("高校导师", academic_draft_review)

        # 企业导师评价初稿
        industry_draft_review = self.industry_advisor.review_paper_draft(initial_draft)
        self.add_to_history("企业导师", industry_draft_review)

        # 博士生撰写相关工作部分
        related_work_prompt = f"""
        基于导师对初稿的反馈：

        高校导师评价：{academic_draft_review}

        企业导师评价：{industry_draft_review}

        请撰写论文的相关工作部分（至少2000字），必须包含：

        1. 系统性的文献分类框架，将相关工作分为3-5个明确的类别
        2. 对每个类别的代表性工作进行深入分析（至少15篇关键文献）
        3. 对每种方法的优缺点进行客观评估，包括：
           - 理论基础的完备性
           - 方法设计的创新性
           - 实验验证的严谨性
           - 应用场景的适用性
        4. 现有研究的局限性和研究空白的系统分析
        5. 本文工作与现有研究的明确区分和创新点

        请确保：
        - 引用最新（过去3年）的相关文献
        - 引用该领域的经典和奠基性工作
        - 客观公正地评价现有工作，避免过度批评
        - 清晰展示研究脉络和发展趋势
        - 为本文的创新点和贡献奠定基础
        """

        related_work_section = self.phd_student.get_response(related_work_prompt)
        self.add_to_history("博士生", f"我撰写的相关工作部分：\n\n{related_work_section}")

        # 博士生撰写方法论部分
        methodology_prompt = f"""
        请撰写论文的方法论部分（至少3000字），必须包含：

        1. 问题的形式化定义，包括数学符号和定义
        2. 方法的理论基础和原理（至少500字）
        3. 完整的技术路线和系统架构
        4. 算法的详细描述，包括：
           - 算法伪代码
           - 关键步骤的数学推导
           - 计算复杂度分析
           - 收敛性或正确性证明（如适用）
        5. 与现有方法的理论比较
        6. 实现细节和关键参数设置

        请确保：
        - 理论推导严谨，数学公式正确
        - 算法描述清晰，便于复现
        - 创新点明确，与现有方法的区别清晰
        - 技术路线完整，逻辑严密
        - 适当使用图表辅助说明复杂概念
        """

        methodology_section = self.phd_student.get_response(methodology_prompt)
        self.add_to_history("博士生", f"我撰写的方法论部分：\n\n{methodology_section}")

        # 博士生撰写实验部分
        experiment_prompt = f"""
        请撰写论文的实验部分（至少3000字），必须包含：

        1. 实验设置（至少500字）：
           - 数据集详细描述（来源、规模、特点）
           - 评估指标定义和选择理由
           - 基线方法选择和实现细节
           - 硬件和软件环境
           - 参数设置和调优过程

        2. 主要实验结果（至少1000字）：
           - 与基线方法的全面比较
           - 结果的统计显著性分析
           - 详细的结果表格和可视化图表
           - 结果分析和讨论

        3. 消融实验（至少500字）：
           - 验证各组件的有效性
           - 关键参数的敏感性分析
           - 不同设置下的性能变化

        4. 案例研究（至少500字）：
           - 典型成功案例分析
           - 失败案例分析和原因探讨
           - 在抖音实际场景中的应用效果

        5. 讨论（至少500字）：
           - 实验结果与理论预期的一致性分析
           - 方法的优势和局限性
           - 潜在的改进方向

        请确保：
        - 实验设计严谨，能够验证方法的有效性
        - 结果呈现全面，包括正面和负面结果
        - 分析深入，不仅展示"是什么"，还解释"为什么"
        - 使用适当的统计方法验证结果的可靠性
        - 实验可复现，提供必要的实现细节
        """

        experiment_section = self.phd_student.get_response(experiment_prompt)
        self.add_to_history("博士生", f"我撰写的实验部分：\n\n{experiment_section}")

        # 博士生撰写结论部分
        conclusion_prompt = f"""
        请撰写论文的结论和未来工作部分（至少1000字），必须包含：

        1. 研究总结（至少300字）：
           - 研究问题回顾
           - 方法创新点概述
           - 主要实验结果和发现

        2. 理论贡献（至少300字）：
           - 对领域知识的推进
           - 理论创新的意义
           - 与现有研究的关系

        3. 实践意义（至少200字）：
           - 在抖音等实际应用场景中的价值
           - 潜在的社会和经济影响

        4. 局限性（至少100字）：
           - 方法的适用条件和边界
           - 尚未解决的挑战

        5. 未来工作（至少200字）：
           - 短期改进方向
           - 长期研究展望
           - 潜在的新研究问题

        请确保：
        - 结论与论文其他部分保持一致
        - 客观评估研究的贡献和局限
        - 未来工作具有可行性和研究价值
        - 语言简洁明了，总结性强
        """

        conclusion_section = self.phd_student.get_response(conclusion_prompt)
        self.add_to_history("博士生", f"我撰写的结论部分：\n\n{conclusion_section}")

        # 整合完整论文
        complete_draft_prompt = f"""
        请将以下各部分整合为一篇完整、连贯的学术论文，确保各部分之间逻辑衔接自然，风格一致：

        1. 摘要和引言：
        {initial_draft}

        2. 相关工作：
        {related_work_section}

        3. 方法论：
        {methodology_section}

        4. 实验：
        {experiment_section}

        5. 结论：
        {conclusion_section}

        整合时请注意：
        - 确保章节编号和引用的一致性
        - 添加必要的过渡段落，增强各部分间的连贯性
        - 统一术语和符号使用
        - 确保参考文献格式规范，按字母顺序排列
        - 检查并修正可能的重复内容
        - 确保总字数至少达到10,000字

        最终论文应当是一个完整、严谨、深入的学术作品，能够在顶级会议或期刊发表。
        """

        complete_draft = self.phd_student.get_response(complete_draft_prompt)
        self.add_to_history("博士生", f"我完善后的完整论文草稿：\n\n{complete_draft}")

        # 更新论文草稿
        self.phd_student.update_paper_draft(complete_draft)
        self.paper_drafts.append({
            "version": 2,
            "content": complete_draft,
            "phase": self.current_phase
        })

        # 阶段总结 - 为每个代理创建阶段记忆
        phase_context = f"""
        研究方向: {self.phd_student.research_topic}
        论文草稿:
        {complete_draft[:1000]}... [论文内容过长，此处截断]
        """

        # 在进入下一阶段前，注入之前阶段的记忆
        self.phd_student.inject_memories_to_context(["initialization", "research_execution"])
        self.academic_advisor.inject_memories_to_context(["initialization", "research_execution"])
        self.industry_advisor.inject_memories_to_context(["initialization", "research_execution"])

        # 博士生总结
        phd_summary = self.phd_student.summarize_phase("paper_writing", phase_context)
        self.add_to_history("系统", f"[博士生阶段总结] {phd_summary}")

        # 高校导师总结
        academic_summary = self.academic_advisor.summarize_phase("paper_writing", phase_context)
        self.add_to_history("系统", f"[高校导师阶段总结] {academic_summary}")

        # 企业导师总结
        industry_summary = self.industry_advisor.summarize_phase("paper_writing", phase_context)
        self.add_to_history("系统", f"[企业导师阶段总结] {industry_summary}")

        print("\n----- 论文撰写阶段完成 -----\n")

    def _paper_optimization_phase(self) -> None:
        """
        论文优化阶段的交互流程 - 深入优化论文质量
        """
        print("\n----- 论文优化阶段开始 -----\n")

        # 获取当前论文草稿
        current_draft = self.phd_student.paper_draft

        # 使用论文评估模块进行全面评估
        evaluation_result = self.evaluation_module.evaluate_paper(current_draft)
        self.add_to_history("论文评估模块", f"论文评估结果：\n\n{evaluation_result}")

        # 生成详细改进计划
        improvement_plan = self.evaluation_module.generate_improvement_plan(current_draft, evaluation_result)
        self.add_to_history("论文评估模块", f"论文改进计划：\n\n{improvement_plan}")

        # 博士生咨询高校导师关于理论创新和学术贡献
        theory_question = self.phd_student.ask_question(
            "academic",
            f"""根据评估结果，我需要深化论文的理论贡献。请您从以下几个方面给予具体指导：

            1. 如何强化我的方法的理论基础？是否需要引入更深入的数学模型或形式化证明？
            2. 我的创新点如何与现有理论框架建立更紧密的联系？
            3. 如何更清晰地阐述我的方法在理论上的突破点？
            4. 有哪些理论分析工具或方法可以用来验证我的方法的理论性质？
            5. 如何在保持理论严谨性的同时，使论文更具可读性？
            """
        )
        self.add_to_history("博士生", theory_question)

        academic_theory_advice = self.academic_advisor.answer_question(theory_question)
        self.add_to_history("高校导师", academic_theory_advice)

        # 博士生咨询企业导师关于实验设计和实用价值
        application_value_question = self.phd_student.ask_question(
            "industry",
            f"""根据评估结果，我需要增强论文的实验验证和实用价值。请您从以下几个方面给予具体指导：

            1. 我的实验设计是否足够全面？还需要增加哪些关键实验？
            2. 如何设计更有说服力的对照实验和消融实验？
            3. 如何更有效地展示方法在抖音实际场景中的应用效果和业务价值？
            4. 有哪些实际案例可以用来说明我的方法解决了实际问题？
            5. 如何量化评估我的方法带来的业务提升？需要哪些具体的指标？
            """
        )
        self.add_to_history("博士生", application_value_question)

        industry_value_advice = self.industry_advisor.answer_question(application_value_question)
        self.add_to_history("企业导师", industry_value_advice)

        # 博士生咨询高校导师关于论文结构和表达
        writing_question = self.phd_student.ask_question(
            "academic",
            f"""为了提升论文的整体质量，我需要优化论文的结构和表达。请您从以下几个方面给予具体指导：

            1. 我的论文结构是否合理？如何调整以更好地突出创新点？
            2. 如何改进论文的逻辑流程，使各部分之间的衔接更加自然？
            3. 有哪些学术写作技巧可以提升论文的可读性和说服力？
            4. 如何更有效地使用图表来展示复杂的概念和实验结果？
            5. 在学术表达上有哪些常见问题需要避免？
            """
        )
        self.add_to_history("博士生", writing_question)

        academic_writing_advice = self.academic_advisor.answer_question(writing_question)
        self.add_to_history("高校导师", academic_writing_advice)

        # 博士生咨询企业导师关于潜在的应用场景扩展
        application_extension_question = self.phd_student.ask_question(
            "industry",
            f"""为了增强论文的影响力，我想探讨方法的应用场景扩展。请您从以下几个方面给予具体指导：

            1. 除了当前讨论的应用场景，我的方法还可以应用于哪些其他业务场景？
            2. 如何证明我的方法具有通用性和可迁移性？
            3. 在不同的应用场景中，我的方法可能需要哪些调整或扩展？
            4. 有哪些潜在的商业价值点是我尚未充分挖掘的？
            5. 如何在论文中展示方法的长期价值和未来发展潜力？
            """
        )
        self.add_to_history("博士生", application_extension_question)

        industry_extension_advice = self.industry_advisor.answer_question(application_extension_question)
        self.add_to_history("企业导师", industry_extension_advice)

        # 博士生优化理论部分
        theory_optimization_prompt = f"""
        基于高校导师关于理论创新的建议，请深度优化论文的理论部分（至少3000字）：

        导师建议：{academic_theory_advice}

        请重点优化以下方面：

        1. 理论基础的深化：
           - 增强方法的数学基础和形式化描述
           - 添加必要的定理、引理或性质及其证明
           - 分析方法的理论特性（如收敛性、稳定性、复杂度等）

        2. 创新点的理论阐述：
           - 明确定位创新点在理论谱系中的位置
           - 与现有理论框架建立明确的联系和区别
           - 深入分析创新点的理论意义和贡献

        3. 理论分析的完备性：
           - 添加必要的理论分析工具和方法
           - 提供理论上的正确性或优越性证明
           - 讨论理论的适用条件和边界

        请确保理论部分既有深度又有可读性，使用精确的数学语言，同时提供必要的解释和直觉理解。
        """

        theory_optimization = self.phd_student.get_response(theory_optimization_prompt)
        self.add_to_history("博士生", f"我优化后的理论部分：\n\n{theory_optimization}")

        # 博士生优化实验部分
        experiment_optimization_prompt = f"""
        基于企业导师关于实验设计和实用价值的建议，请深度优化论文的实验部分（至少3000字）：

        导师建议：{industry_value_advice}

        请重点优化以下方面：

        1. 实验设计的完备性：
           - 添加新的对照实验和消融实验
           - 增强实验的统计显著性分析
           - 扩展实验场景和数据集

        2. 实用价值的展示：
           - 添加真实业务场景的案例研究
           - 量化分析方法带来的业务提升
           - 提供部署细节和实施建议

        3. 结果分析的深度：
           - 深入分析成功和失败案例
           - 探讨方法在不同条件下的表现
           - 与理论预期进行对比分析

        请确保实验部分既有科学严谨性，又能清晰展示方法的实际价值。使用适当的图表和表格来呈现结果，并提供深入的分析和讨论。
        """

        experiment_optimization = self.phd_student.get_response(experiment_optimization_prompt)
        self.add_to_history("博士生", f"我优化后的实验部分：\n\n{experiment_optimization}")

        # 博士生优化论文结构和表达
        writing_optimization_prompt = f"""
        基于高校导师关于论文结构和表达的建议，请优化论文的整体结构和表达（针对全文）：

        导师建议：{academic_writing_advice}

        请重点优化以下方面：

        1. 结构优化：
           - 调整章节顺序和层次结构
           - 增强各部分之间的逻辑衔接
           - 确保论点展开的连贯性和完整性

        2. 表达优化：
           - 提升学术语言的精确性和专业性
           - 消除冗余和模糊表述
           - 增强论证的清晰度和说服力

        3. 可视化优化：
           - 改进图表的设计和呈现
           - 确保图表与文本的紧密结合
           - 添加必要的示意图解释复杂概念

        请提供优化建议和具体的修改方案，而不是重写整篇论文。
        """

        writing_optimization = self.phd_student.get_response(writing_optimization_prompt)
        self.add_to_history("博士生", f"我的论文结构和表达优化方案：\n\n{writing_optimization}")

        # 博士生扩展应用场景
        application_extension_prompt = f"""
        基于企业导师关于应用场景扩展的建议，请撰写论文的应用扩展部分（至少1500字）：

        导师建议：{industry_extension_advice}

        请重点包含以下内容：

        1. 方法的通用性分析：
           - 讨论方法的核心机制如何适用于不同场景
           - 分析方法的可迁移性和适应性
           - 提出方法应用的一般性原则

        2. 扩展应用场景：
           - 详细描述2-3个新的应用场景
           - 分析每个场景的特点和挑战
           - 讨论方法在这些场景中的适用性和潜在效果

        3. 商业价值和影响力：
           - 分析方法在各场景中的商业价值
           - 讨论方法对行业的潜在影响
           - 提出长期发展和应用的愿景

        请确保这部分内容既有学术深度，又有实际应用价值，能够展示方法的广泛适用性和长期影响力。
        """

        application_extension = self.phd_student.get_response(application_extension_prompt)
        self.add_to_history("博士生", f"我撰写的应用扩展部分：\n\n{application_extension}")

        # 整合优化后的论文
        integrate_optimization_prompt = f"""
        请将以下优化内容整合到原论文中，形成一篇完整、高质量的学术论文：

        原论文：
        {current_draft}

        理论部分优化：
        {theory_optimization}

        实验部分优化：
        {experiment_optimization}

        结构和表达优化方案：
        {writing_optimization}

        应用扩展部分：
        {application_extension}

        整合时请注意：
        1. 保持论文结构的完整性和连贯性
        2. 确保各部分之间的逻辑衔接自然
        3. 统一术语、符号和写作风格
        4. 避免内容重复或冲突
        5. 确保参考文献的完整性和规范性
        6. 总字数应至少达到15,000字

        最终论文应当是一个完整、严谨、深入的学术作品，既有扎实的理论基础，又有充分的实验验证，同时展示出广泛的应用价值。
        """

        optimized_draft = self.phd_student.get_response(integrate_optimization_prompt)
        self.add_to_history("博士生", f"我优化后的完整论文：\n\n{optimized_draft}")

        # 更新论文草稿
        self.phd_student.update_paper_draft(optimized_draft)
        self.paper_drafts.append({
            "version": 3,
            "content": optimized_draft,
            "phase": self.current_phase
        })

        # 高校导师最终评审
        final_academic_review = self.academic_advisor.review_paper_draft(optimized_draft)
        self.add_to_history("高校导师", final_academic_review)

        # 企业导师最终评审
        final_industry_review = self.industry_advisor.review_paper_draft(optimized_draft)
        self.add_to_history("企业导师", final_industry_review)

        # 博士生进行最后修改
        final_revision_prompt = f"""
        基于两位导师的最终评审意见，请对论文进行最后的修改和完善：

        高校导师评审：{final_academic_review}

        企业导师评审：{final_industry_review}

        请系统性地解决导师指出的所有问题，并进行以下方面的最终优化：

        1. 修正任何理论错误或不准确之处
        2. 完善实验结果的呈现和分析
        3. 增强论文的整体连贯性和可读性
        4. 确保学术贡献和实用价值的平衡
        5. 完善摘要，使其准确反映论文的核心内容和贡献
        6. 检查并完善参考文献

        请提供最终修改版本，确保其达到顶级会议或期刊的发表标准。
        """

        final_revision = self.phd_student.get_response(final_revision_prompt)
        self.add_to_history("博士生", f"我的论文最终修改版：\n\n{final_revision}")

        # 更新最终论文
        self.phd_student.update_paper_draft(final_revision)
        self.paper_drafts.append({
            "version": 4,
            "content": final_revision,
            "phase": self.current_phase
        })

        # 阶段总结 - 为每个代理创建阶段记忆
        phase_context = f"""
        研究方向: {self.phd_student.research_topic}
        优化后的论文:
        {final_revision[:1000]}... [论文内容过长，此处截断]
        """

        # 在进入下一阶段前，注入之前阶段的记忆
        self.phd_student.inject_memories_to_context(["initialization", "research_execution", "paper_writing"])
        self.academic_advisor.inject_memories_to_context(["initialization", "research_execution", "paper_writing"])
        self.industry_advisor.inject_memories_to_context(["initialization", "research_execution", "paper_writing"])

        # 博士生总结
        phd_summary = self.phd_student.summarize_phase("paper_optimization", phase_context)
        self.add_to_history("系统", f"[博士生阶段总结] {phd_summary}")

        # 高校导师总结
        academic_summary = self.academic_advisor.summarize_phase("paper_optimization", phase_context)
        self.add_to_history("系统", f"[高校导师阶段总结] {academic_summary}")

        # 企业导师总结
        industry_summary = self.industry_advisor.summarize_phase("paper_optimization", phase_context)
        self.add_to_history("系统", f"[企业导师阶段总结] {industry_summary}")

        print("\n----- 论文优化阶段完成 -----\n")

    def _paper_finalization_phase(self) -> None:
        """
        论文定稿发表阶段的交互流程
        """
        print("\n----- 论文定稿发表阶段开始 -----\n")

        # 获取最终论文
        final_paper = self.phd_student.paper_draft

        # 博士生请教目标期刊/会议
        venue_question = self.phd_student.ask_question(
            "academic",
            "我的论文已经接近完成，您认为应该投稿到哪些顶级会议或期刊？请推荐几个最适合的发表渠道。"
        )
        self.add_to_history("博士生", venue_question)

        academic_venue_advice = self.academic_advisor.answer_question(venue_question)
        self.add_to_history("高校导师", academic_venue_advice)

        # 确定目标会议
        target_venue_prompt = f"""
        基于高校导师的建议：{academic_venue_advice}

        请选择一个最适合你论文的目标会议或期刊，并说明选择理由。
        """

        target_venue_decision = self.phd_student.get_response(target_venue_prompt)
        self.add_to_history("博士生", target_venue_decision)

        # 提取目标会议名称（简化处理）
        target_venue = target_venue_decision.split("\n")[0] if "\n" in target_venue_decision else target_venue_decision

        # 检查论文是否达到发表标准
        readiness_check = self.evaluation_module.check_publication_readiness(final_paper, target_venue)
        self.add_to_history("论文评估模块", f"发表就绪性检查结果：\n\n{readiness_check}")

        # 博士生进行最终完善
        finalization_prompt = f"""
        基于发表就绪性检查结果：{readiness_check}

        请对论文进行最终完善，确保其达到{target_venue}的发表标准。重点关注评估中指出的关键问题和可能的批评点。

        此外，请调整论文格式以符合{target_venue}的投稿要求。
        """

        final_paper_version = self.phd_student.get_response(finalization_prompt)
        self.add_to_history("博士生", f"论文最终版本：\n\n{final_paper_version}")

        # 更新最终论文
        self.phd_student.update_paper_draft(final_paper_version)
        self.paper_drafts.append({
            "version": 5,
            "content": final_paper_version,
            "phase": self.current_phase
        })

        # 高校导师最终确认
        final_academic_confirmation = self.academic_advisor.get_response(
            f"请对博士生的最终论文版本进行确认，评估其是否达到{target_venue}的发表标准，并给出最后的建议。\n\n{final_paper_version}"
        )
        self.add_to_history("高校导师", final_academic_confirmation)

        # 企业导师最终确认
        final_industry_confirmation = self.industry_advisor.get_response(
            f"请从产业应用角度，对博士生的最终论文版本进行确认，评估其实用价值和影响力，并给出最后的建议。\n\n{final_paper_version}"
        )
        self.add_to_history("企业导师", final_industry_confirmation)

        # 系统总结
        summary = f"""
        恭喜！博士生在两位导师的指导下，成功完成了一篇高水平论文的撰写。

        论文主题：{self.phd_student.research_topic}

        目标发表会议/期刊：{target_venue}

        论文经历了{len(self.paper_drafts)}个版本的迭代完善，从初始的研究方向确定，到文献调研、方法设计、论文撰写和多轮修改，最终形成了一篇兼具学术价值和实用价值的高质量论文。

        高校导师和企业导师的不同视角和专业指导，为论文提供了全面的改进建议，使论文既有坚实的理论基础，又有明确的实际应用场景和价值。

        希望这篇论文能够顺利发表，为LLM-Agent与数据挖掘的交叉领域做出贡献！
        """

        self.add_to_history("系统", summary)

        # 阶段总结 - 为每个代理创建阶段记忆
        phase_context = f"""
        研究方向: {self.phd_student.research_topic}
        最终论文:
        {final_paper_version[:1000]}... [论文内容过长，此处截断]
        目标发表会议/期刊: {target_venue}
        高校导师最终确认: {final_academic_confirmation}
        企业导师最终确认: {final_industry_confirmation}
        """

        # 在最终阶段，注入所有之前阶段的记忆
        self.phd_student.inject_memories_to_context(["initialization", "research_execution", "paper_writing", "paper_optimization"])
        self.academic_advisor.inject_memories_to_context(["initialization", "research_execution", "paper_writing", "paper_optimization"])
        self.industry_advisor.inject_memories_to_context(["initialization", "research_execution", "paper_writing", "paper_optimization"])

        # 博士生总结
        phd_summary = self.phd_student.summarize_phase("paper_finalization", phase_context)
        self.add_to_history("系统", f"[博士生阶段总结] {phd_summary}")

        # 高校导师总结
        academic_summary = self.academic_advisor.summarize_phase("paper_finalization", phase_context)
        self.add_to_history("系统", f"[高校导师阶段总结] {academic_summary}")

        # 企业导师总结
        industry_summary = self.industry_advisor.summarize_phase("paper_finalization", phase_context)
        self.add_to_history("系统", f"[企业导师阶段总结] {industry_summary}")

        # 整个项目的最终总结
        final_project_summary_prompt = f"""
        请对整个博士论文项目进行全面总结，从研究方向确定到最终论文完成。

        请包含以下内容：
        1. 研究方向的演变和明确过程
        2. 主要研究成果和创新点
        3. 论文的核心贡献
        4. 整个过程中的关键决策和转折点
        5. 对未来研究的启示

        这个总结将作为整个项目的最终记录，应该全面而深入地反映整个研究过程。
        """

        # 博士生提供最终项目总结
        final_project_summary = self.phd_student.get_response(final_project_summary_prompt)
        self.add_to_history("博士生", f"项目最终总结：\n\n{final_project_summary}")

        # 将最终总结也存储为记忆
        self.phd_student.create_memory("project_summary", final_project_summary)

        print("\n----- 论文定稿发表阶段完成 -----\n")

    def save_interaction_history(self, filename: str = "interaction_history.json") -> None:
        """
        保存交互历史到文件

        Args:
            filename: 输出文件名
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.interaction_history, f, ensure_ascii=False, indent=2)

        print(f"交互历史已保存到 {filename}")

    def save_paper_drafts(self, filename: str = "paper_drafts.json") -> None:
        """
        保存论文草稿历史到文件

        Args:
            filename: 输出文件名
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.paper_drafts, f, ensure_ascii=False, indent=2)

        print(f"论文草稿历史已保存到 {filename}")

    def save_agent_memories(self, directory: str = "memories") -> None:
        """
        保存所有代理的记忆到文件

        Args:
            directory: 输出目录
        """
        # 创建目录（如果不存在）
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存博士生记忆
        phd_memories_path = os.path.join(directory, "phd_student_memories.json")
        with open(phd_memories_path, "w", encoding="utf-8") as f:
            json.dump(self.phd_student.memory_bank, f, ensure_ascii=False, indent=2)

        # 保存高校导师记忆
        academic_memories_path = os.path.join(directory, "academic_advisor_memories.json")
        with open(academic_memories_path, "w", encoding="utf-8") as f:
            json.dump(self.academic_advisor.memory_bank, f, ensure_ascii=False, indent=2)

        # 保存企业导师记忆
        industry_memories_path = os.path.join(directory, "industry_advisor_memories.json")
        with open(industry_memories_path, "w", encoding="utf-8") as f:
            json.dump(self.industry_advisor.memory_bank, f, ensure_ascii=False, indent=2)

        print(f"代理记忆已保存到 {directory} 目录")