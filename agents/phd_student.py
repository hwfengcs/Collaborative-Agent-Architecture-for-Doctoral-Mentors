from agents.base_agent import BaseAgent

class PhDStudentAgent(BaseAgent):
    """
    模拟一位人工智能专业的博士研究生，研究方向为LLM-Agent与数据挖掘的交叉领域
    加强创新思维和批判性思考能力
    """
    def __init__(self):
        # 为博士生定义系统提示词，强化学术严谨性和深度
        system_prompt = """
        You are a methodical, grounded PhD student in artificial intelligence, focusing on the intersection of LLM-Agent and data mining. You possess strong academic rigor, systematic research methods, and a commitment to incremental yet meaningful innovation.

        Your main task is to complete a high-quality academic paper with substantial contributions under the guidance of two advisors, which must meet the standards of top academic conferences or journals.

        Your academic advisor is an expert in the LLM-Agent field, while your industry advisor is a data mining expert from ByteDance's TikTok team.

        Your core academic qualities:
        1. Methodical academic thinking: You always adhere to scientific methodology, building systematically on existing work rather than making speculative leaps
        2. Grounded analytical ability: You can identify specific, well-defined limitations in existing research and propose logical improvements
        3. Systematic research methods: Your research follows strict academic standards, including comprehensive literature reviews, clear problem definitions, rigorous method designs, thorough experimental validation, and in-depth result analysis
        4. Practical critical thinking: You can objectively evaluate the advantages and disadvantages of existing methods and identify genuine, addressable research gaps
        5. Clear academic writing: You can present complex academic ideas in a logical, step-by-step manner that shows the natural progression of thought

        In your research process, you must:
        - Identify specific, well-defined research gaps based on thorough literature review
        - Propose solutions that build incrementally on established theoretical foundations
        - Focus on practical, feasible improvements rather than revolutionary but speculative ideas
        - Design rigorous experimental protocols that directly address your specific research questions
        - Maintain logical coherence throughout your work, avoiding conceptual leaps or unfounded claims

        Your paper must include:
        - At least 10,000 words of substantial content (excluding references)
        - Complete related work review (at least 2,000 words) that shows clear understanding of the field's progression
        - Detailed methodology description (at least 3,000 words) that explains how your work builds on existing approaches
        - Comprehensive experimental design and results analysis (at least 3,000 words) with appropriate statistical validation
        - In-depth discussion and future work outlook (at least 1,500 words) that proposes logical next steps
        - At least 30 high-quality references, mainly from top conferences and journals in the past 5 years

        You should constantly ask yourself:
        - Is my research question specific, well-defined, and addressable?
        - Does my method represent a logical extension of existing approaches?
        - Are my claims proportional to my evidence?
        - Have I avoided speculative leaps or unfounded assertions?
        - Does my paper demonstrate a clear, step-by-step progression of ideas?

        Your goal is to complete a high-quality academic paper that represents a meaningful advancement in the field through careful, methodical research rather than speculative or revolutionary claims. Your innovations should be practical, well-grounded in theory, and validated through rigorous experiments.

        IMPORTANT: While all instructions are in English, you must ALWAYS respond in Simplified Chinese.
        """

        super().__init__(
            role="博士生",
            system_prompt=system_prompt
        )

        # 博士生特有的属性
        self.research_topic = None  # 研究主题
        self.research_plan = None   # 研究计划
        self.paper_draft = None     # 论文草稿
        self.innovation_points = [] # 创新点列表
        self.research_challenges = [] # 研究挑战列表

    def set_research_topic(self, topic: str) -> None:
        """设置研究主题"""
        self.research_topic = topic

    def set_research_plan(self, plan: str) -> None:
        """设置研究计划"""
        self.research_plan = plan

    def update_paper_draft(self, draft: str) -> None:
        """更新论文草稿"""
        self.paper_draft = draft

    def add_innovation_point(self, point: str) -> None:
        """添加创新点"""
        self.innovation_points.append(point)

    def add_research_challenge(self, challenge: str) -> None:
        """添加研究挑战"""
        self.research_challenges.append(challenge)

    def brainstorm_innovations(self, context: str) -> str:
        """
        进行基于文献和理论的创新思考

        Args:
            context: 当前研究上下文

        Returns:
            基于文献的创新思考
        """
        prompt = f"""
        As a rigorous PhD student, please conduct a methodical, grounded academic analysis on the following research direction, focusing on incremental yet meaningful innovation:

        Research context: {context}

        Please first systematically analyze:
        1. Research progress and main methods in this field over the past 3 years (cite at least 10 relevant papers)
        2. Theoretical foundations and technical approaches of existing methods
        3. Specific limitations of existing methods at theoretical or practical levels (with concrete evidence from literature)
        4. Well-defined unsolved problems and technical challenges in this field that are acknowledged by the research community

        Based on the above thorough analysis, propose 1-2 feasible research directions that represent logical next steps in the field's progression. For each direction:
        - Clearly define a focused, well-scoped research problem with demonstrable academic value
        - Establish a solid theoretical foundation based directly on existing work, explaining the natural progression from current methods
        - Provide detailed explanation of how your approach addresses specific limitations of existing methods (with direct literature comparisons)
        - Design a practical technical approach that builds incrementally on proven techniques, including detailed algorithmic frameworks or system architectures
        - Formulate verifiable research hypotheses and evaluation metrics that align with established practices in the field
        - Analyze potential technical challenges and provide concrete solution approaches based on existing knowledge

        Your analysis must be firmly grounded in existing literature and established theoretical foundations. Avoid speculative concepts or approaches that lack substantial theoretical support. Each proposed innovation should represent a clear, logical extension of current research with well-defined, achievable contributions.

        Please write in a rigorous academic paper style, emphasizing methodical reasoning, practical feasibility, and incremental advancement rather than revolutionary claims. Include necessary technical details and thorough theoretical analysis that demonstrates the natural evolution from existing work.
        """

        response = self.get_response(prompt, temperature=0.7)  # 使用适中的温度平衡创新性和严谨性

        # 提取创新点并存储
        for point in response.split("\n\n"):
            if point.strip() and ("研究方向" in point or "创新点" in point):
                self.add_innovation_point(point.strip())

        return response

    def critique_existing_approaches(self, approaches: str) -> str:
        """
        系统性批判分析现有方法

        Args:
            approaches: 现有方法描述

        Returns:
            系统性批判分析
        """
        prompt = f"""
        As a PhD student with methodical academic training, please conduct a balanced, evidence-based analysis of the following existing methods, focusing on specific limitations rather than broad criticisms:

        Existing methods: {approaches}

        Please analyze according to the following academic framework:

        1. Theoretical Foundation Analysis
           - Evaluate the specific scope and boundary conditions where theoretical assumptions hold
           - Analyze the mathematical formulations and their appropriateness for the stated problems
           - Identify concrete, demonstrable limitations in theoretical derivations (not speculative ones)
           - Cite relevant literature to support your analysis (at least 5 papers), focusing on empirical evidence

        2. Algorithm/Method Analysis
           - Evaluate the practical implementation aspects of algorithm design
           - Analyze computational efficiency in realistic deployment scenarios
           - Discuss specific convergence properties under different conditions
           - Identify well-documented limitations in particular application contexts

        3. Experimental Evaluation Analysis
           - Evaluate the reproducibility and robustness of experimental designs
           - Analyze the alignment between evaluation metrics and real-world performance needs
           - Discuss confidence intervals and statistical validity of reported results
           - Identify specific experimental scenarios that would provide additional validation

        4. Application Value Analysis
           - Evaluate documented deployment experiences in practical applications
           - Analyze specific engineering challenges with quantifiable impacts
           - Discuss concrete interpretability issues with practical implications
           - Identify specific gaps between laboratory performance and production requirements

        For each analysis dimension, please:
        - Base arguments on documented evidence and published results, avoiding speculation
        - Provide specific, measurable evidence rather than general critiques
        - Cite relevant research that demonstrates the specific limitations you identify
        - Propose incremental, feasible improvement directions that address specific limitations

        Please maintain a balanced perspective that acknowledges both strengths and limitations. Focus on specific, addressable issues rather than fundamental criticisms. Your analysis should identify concrete opportunities for incremental improvement rather than suggesting revolutionary alternatives.
        """

        response = self.get_response(prompt, temperature=0.7)

        # 提取研究挑战并存储
        for challenge in response.split("\n\n"):
            if challenge.strip() and any(keyword in challenge for keyword in ["局限性", "问题", "挑战", "分析"]):
                self.add_research_challenge(challenge.strip())

        return response

    def synthesize_cross_domain_insights(self, domains: str) -> str:
        """
        系统性跨领域知识整合与分析

        Args:
            domains: 相关领域描述

        Returns:
            系统性跨领域分析与研究方向
        """
        prompt = f"""
        As a PhD student with a methodical academic approach, please conduct a focused, evidence-based analysis of practical integration opportunities between the following domains, emphasizing established connections rather than speculative possibilities:

        Related domains: {domains}

        Please analyze according to the following academic framework:

        1. Identification of Established Cross-Domain Connections (at least 1500 words)
           - Review documented examples of successful integration between these domains
           - Analyze specific theoretical concepts that have demonstrably transferred between domains
           - Identify concrete, evidence-based complementarities at the theoretical level
           - Cite published literature demonstrating existing cross-domain work (at least 5 papers per domain)

        2. Analysis of Compatible Methodological Approaches (at least 1500 words)
           - Analyze methodological similarities that facilitate practical integration
           - Evaluate specific technical tools and frameworks used successfully in both domains
           - Identify documented examples of method transfer between domains
           - Discuss practical implementation considerations for method integration

        3. Examination of Shared Research Challenges (at least 1500 words)
           - Analyze specific problems that both domains are actively addressing
           - Evaluate documented cases where one domain has informed solutions in the other
           - Identify concrete problem areas where combined approaches have shown promise
           - Discuss practical challenges in implementing cross-domain collaborations

        4. Focused Cross-domain Research Direction Proposals (at least 2000 words)
           - Propose 1-2 well-defined, practical cross-domain research directions based on established connections
           - For each direction, elaborate in detail:
             * Specific, narrowly-scoped research problem with documented relevance to both domains
             * Evidence-based theoretical foundation drawing on successful prior integrations
             * Step-by-step method design that builds incrementally on established approaches
             * Realistic expected contributions with appropriate scope
             * Specific implementation challenges and practical mitigation strategies
             * Concrete validation plans using established evaluation frameworks

        Your analysis must:
        - Be based on documented evidence of existing cross-domain work
        - Focus on practical, implementable integration opportunities
        - Cite relevant literature demonstrating successful precedents (at least 20 papers)
        - Maintain logical progression and appropriate scope of claims

        Please write in a clear, structured academic style that emphasizes concrete connections and practical integration opportunities. Your goal is to propose cross-domain research directions that represent logical next steps building on established foundations, not speculative leaps.
        """

        return self.get_response(prompt, temperature=0.7)

    def ask_question(self, advisor_type: str, question: str) -> str:
        """
        记录向导师提出的问题，返回格式化的问题

        Args:
            advisor_type: 导师类型 ("academic" 或 "industry")
            question: 问题内容

        Returns:
            格式化的问题
        """
        advisor_title = "高校导师" if advisor_type == "academic" else "企业导师"
        formatted_question = f"[博士生向{advisor_title}提问] {question}"
        return formatted_question