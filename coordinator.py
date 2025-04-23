from agents.phd_student import PhDStudentAgent
from agents.academic_advisor import AcademicAdvisorAgent
from agents.industry_advisor import IndustryAdvisorAgent
from modules.knowledge_retrieval import KnowledgeRetrievalModule
from modules.paper_evaluation import PaperEvaluationModule
import json
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
        
        print("\n----- 研究执行阶段完成 -----\n")
    
    def _paper_writing_phase(self) -> None:
        """
        论文撰写阶段的交互流程
        """
        print("\n----- 论文撰写阶段开始 -----\n")
        
        # 博士生请教论文结构
        structure_question = self.phd_student.ask_question(
            "academic",
            "我准备开始撰写论文，对于我们的研究方向，论文应该如何组织结构才能清晰地表达研究成果？"
        )
        self.add_to_history("博士生", structure_question)
        
        academic_structure_advice = self.academic_advisor.answer_question(structure_question)
        self.add_to_history("高校导师", academic_structure_advice)
        
        # 博士生请教实验设计
        experiment_question = self.phd_student.ask_question(
            "industry",
            "对于我们的研究，应该设计哪些实验来验证方法的有效性？尤其是在抖音这样的实际场景中，如何设计有说服力的评估指标？"
        )
        self.add_to_history("博士生", experiment_question)
        
        industry_experiment_advice = self.industry_advisor.answer_question(experiment_question)
        self.add_to_history("企业导师", industry_experiment_advice)
        
        # 博士生撰写论文初稿
        paper_draft_prompt = f"""
        请基于我们的研究方向、文献调研结果以及导师的建议，撰写一篇论文的摘要和引言部分。
        
        研究方向：{self.phd_student.research_topic}
        
        论文结构参考：{academic_structure_advice}
        
        实验设计参考：{industry_experiment_advice}
        
        请撰写：
        1. 摘要（不超过250字）
        2. 引言（包括研究背景、问题陈述、研究意义、主要贡献等）
        
        请确保内容既有学术深度，又突出实际应用价值。
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
        
        # 博士生完善论文
        complete_draft_prompt = f"""
        基于导师对初稿的反馈：
        
        高校导师评价：{academic_draft_review}
        
        企业导师评价：{industry_draft_review}
        
        请完善论文，添加以下部分：
        
        1. 相关工作
        2. 方法论
        3. 实验设计
        4. 结果与讨论
        5. 结论
        
        请确保论文结构完整，逻辑清晰，并充分考虑两位导师的反馈意见。
        """
        
        complete_draft = self.phd_student.get_response(complete_draft_prompt)
        self.add_to_history("博士生", f"我完善后的论文草稿：\n\n{complete_draft}")
        
        # 更新论文草稿
        self.phd_student.update_paper_draft(complete_draft)
        self.paper_drafts.append({
            "version": 2,
            "content": complete_draft,
            "phase": self.current_phase
        })
        
        print("\n----- 论文撰写阶段完成 -----\n")
    
    def _paper_optimization_phase(self) -> None:
        """
        论文优化阶段的交互流程
        """
        print("\n----- 论文优化阶段开始 -----\n")
        
        # 获取当前论文草稿
        current_draft = self.phd_student.paper_draft
        
        # 使用论文评估模块评估论文质量
        evaluation_result = self.evaluation_module.evaluate_paper(current_draft)
        self.add_to_history("论文评估模块", f"论文评估结果：\n\n{evaluation_result}")
        
        # 生成改进计划
        improvement_plan = self.evaluation_module.generate_improvement_plan(current_draft, evaluation_result)
        self.add_to_history("论文评估模块", f"论文改进计划：\n\n{improvement_plan}")
        
        # 博士生咨询高校导师关于理论创新
        theory_question = self.phd_student.ask_question(
            "academic",
            f"根据评估结果，我的论文在理论创新性方面还有提升空间。您对如何增强论文的理论贡献有什么建议？"
        )
        self.add_to_history("博士生", theory_question)
        
        academic_theory_advice = self.academic_advisor.answer_question(theory_question)
        self.add_to_history("高校导师", academic_theory_advice)
        
        # 博士生咨询企业导师关于实用价值
        application_value_question = self.phd_student.ask_question(
            "industry",
            f"评估指出我的论文在实用价值方面有待加强。您能给我一些建议，如何更好地展示研究成果在抖音等实际场景中的应用价值和效果？"
        )
        self.add_to_history("博士生", application_value_question)
        
        industry_value_advice = self.industry_advisor.answer_question(application_value_question)
        self.add_to_history("企业导师", industry_value_advice)
        
        # 博士生优化论文
        optimize_draft_prompt = f"""
        基于论文评估结果和导师的建议，请优化你的论文：
        
        评估结果：{evaluation_result}
        
        改进计划：{improvement_plan}
        
        高校导师关于理论创新的建议：{academic_theory_advice}
        
        企业导师关于实用价值的建议：{industry_value_advice}
        
        请重点改进以下方面：
        1. 增强理论创新性
        2. 提升实用价值展示
        3. 完善实验评估
        4. 优化论文结构和表达
        
        请提供优化后的完整论文。
        """
        
        optimized_draft = self.phd_student.get_response(optimize_draft_prompt)
        self.add_to_history("博士生", f"我优化后的论文：\n\n{optimized_draft}")
        
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
        基于两位导师的最终评审意见：
        
        高校导师评审：{final_academic_review}
        
        企业导师评审：{final_industry_review}
        
        请对论文进行最后的修改和完善，注意平衡学术价值和实用价值。
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