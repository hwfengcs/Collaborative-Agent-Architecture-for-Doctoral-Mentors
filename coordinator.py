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
            "作为一名人工智能专业的博士生，我需要确定一个研究方向。请思考一个在LLM-Agent与互联网内容挖掘交叉领域的有价值研究方向，特别关注实际应用场景和可量化的业务价值。研究方向应该解决互联网内容平台（如抖音）面临的实际数据挖掘挑战，并能够产生明确的业务指标改进。"
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

        请整合两位导师的意见，确定一个明确的研究方向和初步的研究计划。研究方向应该在LLM-Agent与互联网内容挖掘的交叉领域，特别关注：

        1. 解决互联网内容平台（如抖音）面临的实际数据挖掘挑战
        2. 具有明确的业务价值和可量化的性能指标
        3. 系统可在真实或类真实的大规模互联网数据上验证
        4. 具备工程可实现性和部署可行性
        5. 既有学术创新价值又有明确的产业应用前景

        请提出一个具体、聚焦的研究方向，而不是泛泛而谈。研究方向应包含明确的问题定义、技术路线、预期贡献和业务价值。
        """

        final_direction = self.phd_student.get_response(direction_prompt)
        self.add_to_history("博士生", final_direction)

        # 保存研究方向
        self.phd_student.set_research_topic(final_direction)

        # 博士生制定研究计划
        research_plan_prompt = """
        基于确定的研究方向，请制定一个详细的研究计划，特别关注互联网内容挖掘的实际应用价值和系统实现，包括：

        1. 研究问题和业务目标：
           - 明确定义要解决的互联网内容挖掘问题
           - 设定具体、可量化的业务指标和技术目标
           - 分析问题的实际业务价值和技术挑战

        2. 文献与行业实践调研计划：
           - 学术文献调研重点和方法
           - 工业界技术报告和系统论文调研
           - 开源工具和框架调研
           - 互联网公司实践经验调研方法

        3. 系统设计与技术路线：
           - 整体系统架构设计
           - 核心算法和技术选择
           - 数据处理流程和特征工程
           - 工程实现策略和技术栈选择
           - 与现有系统的集成方案

        4. 实验与评估计划：
           - 数据集选择和准备（优先考虑真实或类真实互联网数据）
           - 离线实验设计和评估指标
           - 在线或模拟在线实验设计
           - A/B测试方案和业务指标评估
           - 系统性能和资源效率测试

        5. 预期成果与业务价值：
           - 技术创新点和学术贡献
           - 预期业务指标改进
           - 系统部署和应用场景
           - 潜在的产品化和商业化路径

        6. 资源需求与时间规划：
           - 数据和计算资源需求
           - 人力资源和专业技能需求
           - 详细的时间里程碑和交付物
           - 风险评估和应对策略

        请尽可能详细地描述每个部分，确保研究计划既有学术严谨性，又有明确的工程实现路径和业务价值。
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

        请全面修改你的研究计划，重点回应导师的建议和关注点，特别是：

        1. 确保研究问题更加聚焦于互联网内容挖掘的实际挑战
        2. 增强系统设计的工程实现细节和可行性分析
        3. 完善实验评估方案，特别是业务指标的量化评估方法
        4. 加强与现有工业界系统和实践的对比分析
        5. 提供更详细的系统部署和应用场景分析
        6. 明确技术创新点如何转化为实际业务价值
        7. 完善风险评估和应对策略，特别是工程实现风险

        请确保修改后的研究计划既满足学术严谨性要求，又具有明确的工程实现路径和业务价值。对每个修改点，请明确说明如何回应导师的具体建议。
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

        knowledge_response = self.knowledge_module.consult_llm(knowledge_query)
        # 直接添加回答到历史
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
        请基于我们的研究方向、文献调研结果以及导师的建议，撰写一篇高质量学术论文的摘要和引言部分，特别强调互联网内容挖掘的实际应用价值和可量化的业务效果。

        研究方向：{self.phd_student.research_topic}

        理论框架与结构参考：{academic_structure_advice}

        实验设计参考：{industry_experiment_advice}

        文献综述方法：{academic_literature_advice}

        请撰写：
        1. 摘要（250-300字）：必须包含以下要素：
           - 互联网数据挖掘中的实际业务问题和挑战
           - 明确的问题定义和业务价值
           - 方法的技术创新点和系统架构特点
           - 在真实或类真实大规模数据集上的实验结果
           - 可量化的业务指标改进
           - 实际部署场景和应用价值

        2. 引言（至少1800字）：必须包含以下要素：
           - 互联网内容挖掘的业务背景与产业价值（至少300字）
           - 现有方法在实际应用中的系统性分析与工程局限性（至少500字）
           - 明确的工程问题定义与技术挑战（至少250字）
           - 本文方法的技术基础与工程创新点（至少350字）
           - 系统架构和实现方案概述（至少200字）
           - 主要贡献的详细阐述，包括技术和业务两方面（至少250字）
           - 论文结构概述

        请确保内容：
        - 学术语言严谨，同时注重工程实用性
        - 每个观点都有实证支持或来自实际系统的证据
        - 清晰界定应用场景和系统要求
        - 准确引用相关文献（至少10篇，优先引用有实际应用验证的工作）
        - 包含具体的技术细节和系统设计考量
        - 讨论实际部署中的工程挑战和解决方案
        - 提供明确的业务价值和性能指标
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

        请撰写论文的相关工作部分（至少2500字），特别关注互联网内容挖掘领域的实际应用系统和工业界实践，必须包含：

        1. 系统性的文献分类框架，将相关工作分为4-6个明确的类别，其中至少包含：
           - 学术研究方法
           - 工业界实际部署系统
           - 开源工具和框架
           - 数据集和评估基准

        2. 对每个类别的代表性工作进行深入分析（至少20篇关键文献，其中至少8篇来自工业界技术报告或系统论文）：
           - 学术论文分析（至少10篇）
           - 工业界技术报告和系统论文（至少8篇）
           - 开源项目和工具文档（至少2个）

        3. 对每种方法进行全面评估，特别关注实际应用价值：
           - 系统架构和技术栈
           - 工程实现的复杂度和可维护性
           - 计算资源需求和性能特性
           - 在大规模数据上的可扩展性
           - 实际部署环境和应用场景
           - 报告的业务指标改进
           - 工程挑战和解决方案

        4. 现有系统在实际互联网应用中的局限性分析：
           - 技术局限性（算法、架构等）
           - 工程局限性（可扩展性、维护性等）
           - 业务局限性（适用场景、效果边界等）
           - 数据局限性（数据质量、隐私等）

        5. 本文工作与现有系统的明确区分和创新点，特别是在实际应用价值方面

        请确保：
        - 引用最新（过去2年）的工业界实践和系统论文
        - 引用主要互联网公司（如Google、Meta、ByteDance等）的技术博客和系统论文
        - 引用该领域的经典系统和开源框架
        - 分析实际部署系统的架构、性能和业务价值
        - 客观评估现有系统的工程实现难度和维护成本
        - 清晰展示技术演进路线和工业应用趋势
        - 为本文的工程创新点和业务价值奠定基础
        """

        related_work_section = self.phd_student.get_response(related_work_prompt)
        self.add_to_history("博士生", f"我撰写的相关工作部分：\n\n{related_work_section}")

        # 博士生撰写方法论部分
        methodology_prompt = f"""
        请撰写论文的方法论部分（至少3500字），特别强调系统设计、工程实现和互联网应用场景，必须包含：

        1. 问题的工程化定义，包括：
           - 业务需求和技术挑战
           - 系统输入输出规范
           - 性能指标和约束条件
           - 形式化定义和数学符号（如适用）

        2. 系统架构和技术栈（至少600字）：
           - 整体架构图及组件说明
           - 数据流和处理流程
           - 各模块的功能和接口定义
           - 技术栈选择理由和优势
           - 与现有系统的集成方案

        3. 核心算法和方法（至少800字）：
           - 算法伪代码或流程图
           - 关键步骤的详细说明
           - 计算复杂度和空间复杂度分析
           - 优化策略和技术创新点
           - 与现有方法的对比分析

        4. 工程实现细节（至少800字）：
           - 数据预处理和特征工程
           - 分布式计算和并行处理策略
           - 缓存机制和性能优化
           - 关键参数设置和调优方法
           - 异常处理和容错机制
           - 监控和日志系统

        5. 扩展性和可维护性设计（至少500字）：
           - 系统扩展策略（水平/垂直扩展）
           - 模块化设计和代码组织
           - 配置管理和版本控制
           - 部署流程和环境要求
           - 维护和更新机制

        6. 实际应用场景适配（至少400字）：
           - 针对不同业务场景的适配策略
           - 冷启动和数据稀疏问题解决方案
           - A/B测试和效果评估方法
           - 业务规则集成和策略配置

        7. 与现有系统的比较（至少400字）：
           - 技术路线对比
           - 工程复杂度对比
           - 资源需求对比
           - 可扩展性对比
           - 业务适用性对比

        请确保：
        - 系统设计合理，架构图清晰完整
        - 工程实现细节充分，便于实际部署
        - 算法描述准确，伪代码或流程图完整
        - 创新点明确，与现有系统的优势清晰
        - 技术路线完整，逻辑严密
        - 充分考虑互联网场景的特殊需求（如高并发、大数据量、实时性等）
        - 适当使用图表辅助说明系统架构和数据流
        - 讨论实际部署中可能面临的挑战和解决方案
        """

        methodology_section = self.phd_student.get_response(methodology_prompt)
        self.add_to_history("博士生", f"我撰写的方法论部分：\n\n{methodology_section}")

        # 博士生撰写实验部分
        experiment_prompt = f"""
        请撰写论文的实验部分（至少3500字），特别强调在真实或类真实互联网数据集上的性能表现和业务指标改进，必须包含：

        1. 实验设置与评估框架（至少700字）：
           - 数据集详细描述：
             * 真实业务数据集（如可能）或公开大规模互联网数据集
             * 数据规模、特征分布和业务特点
             * 数据预处理和特征工程流程
             * 训练/验证/测试集划分策略
           - 评估指标体系：
             * 技术指标（准确率、召回率、F1等）
             * 业务指标（点击率、转化率、用户留存等）
             * 系统指标（延迟、吞吐量、资源利用率等）
           - 基线系统选择：
             * 学术基线方法
             * 工业界现有系统
             * 开源解决方案
           - 实验环境：
             * 硬件配置（CPU、内存、GPU等）
             * 软件环境（框架、库、版本等）
             * 分布式设置（如适用）
           - 超参数设置和调优策略

        2. 离线实验结果与分析（至少1000字）：
           - 与基线系统的全面对比：
             * 各项技术指标的详细比较
             * 不同数据规模下的性能变化
             * 不同业务场景下的表现差异
           - 性能分析：
             * 统计显著性检验
             * 结果可靠性和稳定性分析
             * 详细的结果表格和可视化图表
           - 系统效率分析：
             * 计算资源消耗对比
             * 训练和推理时间对比
             * 存储需求对比

        3. 在线A/B测试或模拟在线评估（至少600字）：
           - 实验设置：
             * 流量分配策略
             * 实验周期和样本量
             * 监控指标和告警机制
           - 业务指标改进：
             * 核心业务KPI的提升
             * 用户体验指标的变化
             * 长期效果与短期效果对比
           - 系统稳定性评估：
             * 峰值负载处理能力
             * 错误率和恢复机制
             * 资源利用效率

        4. 消融实验与组件分析（至少600字）：
           - 系统各组件的贡献度量：
             * 关键模块的独立效果评估
             * 组件间协同效应分析
           - 关键设计选择的验证：
             * 架构设计选择的影响
             * 算法选择的影响
             * 特征工程策略的影响
           - 参数敏感性分析：
             * 关键参数的影响范围
             * 最优参数区间的确定
             * 参数调整的业务影响

        5. 真实业务案例分析（至少600字）：
           - 成功案例详细分析：
             * 具体业务场景描述
             * 系统表现和业务价值
             * 关键成功因素分析
           - 挑战案例分析：
             * 系统表现不佳的场景
             * 原因分析和解决方案
             * 经验教训和改进方向
           - 部署经验总结：
             * 实际部署流程和挑战
             * 监控和维护经验
             * 用户反馈和迭代优化

        请确保：
        - 实验设计全面，能够验证系统在真实互联网环境中的有效性
        - 评估指标体系完整，同时覆盖技术指标和业务指标
        - 结果呈现客观，包括优势和局限性
        - 分析深入，解释性能差异的原因和业务影响
        - 使用适当的统计方法验证结果的可靠性和显著性
        - 讨论系统在不同业务场景和数据条件下的适应性
        - 提供足够的实现细节，确保系统可复现
        - 强调系统的实际业务价值和部署经验
        """

        experiment_section = self.phd_student.get_response(experiment_prompt)
        self.add_to_history("博士生", f"我撰写的实验部分：\n\n{experiment_section}")

        # 博士生撰写结论部分
        conclusion_prompt = f"""
        请撰写论文的结论和未来工作部分（至少1500字），特别强调系统的实际应用价值、业务影响和产业化前景，必须包含：

        1. 研究与工程成果总结（至少400字）：
           - 所解决的实际互联网数据挖掘问题回顾
           - 系统架构和技术创新点概述
           - 主要实验结果和业务指标改进
           - 实际部署情况和用户反馈

        2. 技术贡献（至少300字）：
           - 系统设计和工程实现的创新点
           - 算法和架构优化的技术价值
           - 与现有系统的技术对比和优势
           - 开源组件和工具的贡献（如适用）

        3. 业务价值与产业影响（至少400字）：
           - 在互联网内容挖掘中的具体业务价值
           - 量化的业务指标改进和经济效益
           - 用户体验提升和业务流程优化
           - 对互联网产业的潜在影响和启示
           - 技术推广和产业化路径

        4. 系统局限性与工程挑战（至少200字）：
           - 当前系统的技术局限和适用条件
           - 规模化部署面临的工程挑战
           - 数据质量和隐私保护问题
           - 尚未解决的业务需求和技术难点

        5. 未来工作与演进路线（至少300字）：
           - 短期系统优化和功能扩展计划
           - 中期技术迭代和架构演进方向
           - 长期研究方向和技术展望
           - 跨场景应用扩展和商业化前景
           - 具体的技术路线图和里程碑

        请确保：
        - 结论与论文其他部分保持一致，特别是与实验结果和业务案例分析
        - 客观评估系统的技术贡献和业务价值，提供具体的量化指标
        - 坦诚讨论系统的局限性和面临的挑战，以及相应的解决思路
        - 未来工作具有明确的技术路线和业务价值，不是泛泛而谈
        - 强调系统在互联网内容挖掘领域的实际应用前景和产业化潜力
        - 语言既有学术严谨性，又具备工程实用性和业务洞察
        """

        conclusion_section = self.phd_student.get_response(conclusion_prompt)
        self.add_to_history("博士生", f"我撰写的结论部分：\n\n{conclusion_section}")

        # 整合完整论文
        complete_draft_prompt = f"""
        请将以下各部分整合为一篇完整、连贯的学术论文，特别强调互联网内容挖掘的实际应用价值和系统实现，确保各部分之间逻辑衔接自然，风格一致：

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

        整合时请特别注意：
        - 确保论文整体突出互联网内容挖掘的实际应用价值和业务效果
        - 强调系统设计、工程实现和部署经验的详细描述
        - 保持技术细节与业务价值的平衡，既有学术深度又有工程实用性
        - 确保章节编号和引用的一致性，特别是工业界技术报告和系统论文的引用
        - 添加必要的过渡段落，增强各部分间的连贯性和论证的完整性
        - 统一术语、符号和技术概念的使用，确保专业准确性
        - 确保系统架构图、数据流程图和实验结果图表的清晰和一致性
        - 检查并修正可能的重复内容，确保论述精炼
        - 确保参考文献格式规范，按字母顺序排列，同时包含学术论文和工业界技术报告
        - 确保总字数至少达到12,000字，其中系统设计和实验部分应占较大比重

        最终论文应当是一个完整、严谨、深入的学术作品，同时具有明确的工程实用价值和产业应用前景，能够在数据挖掘和信息检索领域的顶级会议（如KDD、WWW、SIGIR等）发表，并对互联网企业的实际业务有参考价值。
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
        基于高校导师关于理论创新的建议，请深度优化论文的理论部分（至少3000字），特别强调理论与工程实践的结合以及在互联网内容挖掘中的应用价值：

        导师建议：{academic_theory_advice}

        请重点优化以下方面：

        1. 理论基础与工程实践的结合：
           - 增强方法的数学基础和形式化描述，同时关联实际系统实现
           - 分析理论模型在工程实现中的简化和适应性调整
           - 讨论理论保证与工程约束的平衡策略
           - 分析方法的理论特性（如收敛性、稳定性、复杂度等）对系统性能的影响

        2. 创新点的理论与实用价值阐述：
           - 明确定位创新点在理论谱系和工业实践中的位置
           - 与现有理论框架和工业系统建立明确的联系和区别
           - 深入分析创新点的理论意义和实际业务价值
           - 讨论理论创新如何解决实际互联网数据挖掘中的关键挑战

        3. 理论分析的工程适用性：
           - 添加必要的理论分析工具和方法，并讨论其工程实现
           - 提供理论上的正确性或优越性证明，以及在实际数据上的验证
           - 讨论理论的适用条件和边界，特别是在互联网规模数据上的适用性
           - 分析理论模型在不同业务场景中的适应性和泛化能力
           - 讨论理论简化与工程实现之间的权衡

        4. 理论模型的系统实现指导：
           - 详细阐述理论模型如何指导系统架构设计
           - 分析关键理论参数对系统性能的影响及调优策略
           - 讨论理论保证如何转化为工程实现中的质量保证机制
           - 提供从理论到实践的实现路径和最佳实践

        请确保理论部分既有学术深度又有工程实用性，使用精确的数学语言描述核心理论，同时提供必要的工程解释和实现指导。理论分析应当既能满足学术严谨性要求，又能为工程师提供实用的系统设计和实现指导。
        """

        theory_optimization = self.phd_student.get_response(theory_optimization_prompt)
        self.add_to_history("博士生", f"我优化后的理论部分：\n\n{theory_optimization}")

        # 博士生优化实验部分
        experiment_optimization_prompt = f"""
        基于企业导师关于实验设计和实用价值的建议，请深度优化论文的实验部分（至少3500字），特别强调在真实互联网环境中的业务价值和实际部署效果：

        导师建议：{industry_value_advice}

        请重点优化以下方面：

        1. 实验设计的产业相关性（至少800字）：
           - 添加更多真实或类真实互联网数据集的实验
           - 设计更贴近实际业务场景的评估方法
           - 增加与工业界标准系统的对比实验
           - 设计针对互联网特有挑战的专项实验（如数据稀疏性、冷启动、长尾分布等）
           - 增强实验的统计显著性分析和可靠性验证

        2. 业务价值的量化展示（至少800字）：
           - 添加详细的业务指标分析（如点击率、转化率、用户留存等）
           - 量化分析方法带来的直接经济效益和投资回报率
           - 提供真实或模拟的A/B测试结果及其统计分析
           - 分析系统对不同用户群体和业务场景的影响差异
           - 讨论长期业务价值和短期效果的平衡

        3. 系统性能和资源效率分析（至少600字）：
           - 详细分析系统在不同负载下的性能表现
           - 提供资源消耗（CPU、内存、存储、网络等）的详细测试结果
           - 分析系统的扩展性和成本效益
           - 比较不同部署方案的性能和成本权衡
           - 讨论在资源受限条件下的优化策略

        4. 实际部署案例和经验（至少700字）：
           - 提供详细的系统部署流程和架构图
           - 分析实际部署中遇到的挑战和解决方案
           - 讨论系统监控、维护和迭代优化的经验
           - 提供用户反馈和系统改进的案例分析
           - 分享工程实践中的经验教训和最佳实践

        5. 结果分析的商业洞察（至少600字）：
           - 深入分析系统成功和失败案例背后的业务因素
           - 探讨方法在不同业务条件和用户行为模式下的表现差异
           - 分析系统对业务决策的支持价值
           - 讨论实验结果对产品设计和业务策略的指导意义
           - 提出基于实验结果的业务优化建议

        请确保实验部分既有科学严谨性，又有明确的业务价值和工程实用性。使用丰富的图表和表格来呈现结果，包括业务指标仪表盘、性能监控图表、资源利用率分析等。提供深入的分析和讨论，特别关注系统如何解决实际互联网内容挖掘中的关键业务挑战。
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
        基于企业导师关于应用场景扩展的建议，请撰写论文的应用扩展部分（至少2000字），特别聚焦于互联网内容挖掘的多样化场景和实际业务价值：

        导师建议：{industry_extension_advice}

        请重点包含以下内容：

        1. 系统的跨场景适应性分析（至少500字）：
           - 详细分析系统核心组件在不同互联网内容类型上的适应性（如短视频、长视频、图文、音频等）
           - 讨论系统架构的模块化设计如何支持不同业务场景的快速适配
           - 分析系统在不同数据规模、用户行为模式和业务目标下的可调整性
           - 提出系统迁移到新场景的具体工程实践指南和最佳实践
           - 讨论系统的可配置性和参数调优策略

        2. 互联网内容挖掘的扩展应用场景（至少1000字）：
           - 详细描述3-5个具体的互联网内容挖掘应用场景，包括：
             * 短视频推荐和内容理解
             * 多模态内容分析和跨模态检索
             * 用户兴趣挖掘和个性化服务
             * 内容安全与质量控制
             * 创作者生态与内容生产辅助
           - 对每个场景进行深入分析：
             * 具体业务需求和技术挑战
             * 数据特点和处理要求
             * 系统适配方案和架构调整
             * 预期业务指标和评估方法
             * 实施路径和资源需求

        3. 商业价值和产业影响（至少500字）：
           - 量化分析系统在各场景中的具体商业价值：
             * 直接业务指标改进（如点击率、转化率、用户时长等）
             * 运营效率提升和成本节约
             * 用户体验改善和平台生态健康度
           - 讨论系统对互联网内容产业的潜在影响：
             * 内容生产和分发效率的提升
             * 用户体验和参与度的改变
             * 创作者生态的优化
             * 内容多样性和质量的提升
           - 提出基于系统的创新业务模式和商业化路径
           - 分析系统在国内外互联网内容平台的应用前景和竞争优势

        请确保这部分内容既有技术深度，又有明确的业务场景和商业价值分析。每个应用场景应当包含具体的技术实现方案、业务指标预期和实施建议，而不是泛泛而谈。特别强调系统如何解决互联网内容挖掘中的实际痛点问题，以及如何为内容平台创造可量化的商业价值。
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
        基于两位导师的最终评审意见，请对论文进行最后的修改和完善，特别强调互联网内容挖掘的实际应用价值和系统实现细节：

        高校导师评审：{final_academic_review}

        企业导师评审：{final_industry_review}

        请系统性地解决导师指出的所有问题，并进行以下方面的最终优化：

        1. 技术内容优化：
           - 修正任何理论错误或技术描述不准确之处
           - 确保系统架构描述的完整性和一致性
           - 完善算法和工程实现细节，增强可复现性
           - 确保技术创新点的清晰表达和充分论证

        2. 实验与业务价值优化：
           - 完善实验结果的呈现和分析，特别是业务指标的量化展示
           - 增强A/B测试或在线评估结果的可信度和统计显著性
           - 深化系统性能和资源效率分析
           - 强化真实业务案例的详细描述和价值分析

        3. 工程实践与部署经验优化：
           - 增加系统实际部署的具体细节和最佳实践
           - 完善系统监控、维护和迭代优化的经验分享
           - 详细讨论系统在不同业务场景的适配策略
           - 提供更具操作性的工程实施指南

        4. 结构与表达优化：
           - 增强论文的整体连贯性和逻辑流畅度
           - 确保学术贡献和工程实用价值的平衡展示
           - 优化图表和表格的呈现，使其更直观清晰
           - 统一术语和概念使用，确保专业准确性

        5. 摘要与引言优化：
           - 完善摘要，确保其准确反映系统的核心技术创新和业务价值
           - 强化引言中对互联网内容挖掘实际挑战的描述
           - 明确突出系统的主要贡献和实际应用价值

        6. 参考文献与学术规范：
           - 检查并完善参考文献，确保包含最新的工业界技术报告和系统论文
           - 确保引用格式的一致性和完整性
           - 适当增加对工业界实践的引用和讨论

        请提供最终修改版本，确保其既达到数据挖掘和信息检索领域顶级会议（如KDD、WWW、SIGIR等）的学术标准，又具有明确的工程实用价值和产业应用指导意义，能够为互联网内容平台的技术团队提供实际参考。
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