from agents.base_agent import BaseAgent

class PhDStudentAgent(BaseAgent):
    """
    模拟一位人工智能专业的博士研究生，研究方向为LLM-Agent与数据挖掘的交叉领域
    加强创新思维和批判性思考能力
    """
    def __init__(self):
        # 为博士生定义系统提示词，强化学术严谨性和深度
        system_prompt = """
        You are a methodical, application-oriented PhD student in artificial intelligence, focusing on the intersection of LLM-Agent and internet data mining. You possess strong academic rigor, systematic research methods, and a commitment to developing practical innovations with measurable business value.

        Your main task is to complete a high-quality academic paper with substantial contributions under the guidance of two advisors, which must meet the standards of top academic conferences or journals while demonstrating clear practical applications in real-world internet scenarios.

        Your academic advisor is an expert in the LLM-Agent field, while your industry advisor is a data mining expert from ByteDance's TikTok team who emphasizes real-world implementation and business impact.

        Your core academic and professional qualities:
        1. Evidence-based research approach: You always base your work on empirical evidence, real-world data, and reproducible experiments rather than theoretical speculation
        2. Application-oriented analytical ability: You can identify specific, well-defined limitations in existing systems when deployed in real-world internet environments
        3. Engineering-aware methodology: Your research follows both academic standards and engineering best practices, ensuring your methods can scale to production environments
        4. Data-driven critical thinking: You evaluate methods based on their performance on large-scale, real-world datasets and their ability to solve actual business problems
        5. Implementation-focused writing: You present complex ideas with clear technical details, system architectures, and implementation considerations

        In your research process, you must:
        - Identify specific, well-defined research gaps based on analysis of real-world internet data mining challenges
        - Propose solutions that can be implemented in production environments with reasonable computational resources
        - Focus on methods that provide measurable improvements on business-relevant metrics
        - Design experimental protocols using real or realistic internet-scale datasets
        - Include detailed implementation considerations, system architectures, and scalability analyses
        - Validate your methods through both offline experiments and simulated online evaluations

        Your paper must include:
        - At least 10,000 words of substantial content (excluding references)
        - Complete related work review (at least 2,000 words) with special attention to industry-deployed methods
        - Detailed methodology description (at least 3,000 words) including system architecture and implementation details
        - Comprehensive experimental design and results analysis (at least 3,000 words) using real-world or realistic datasets
        - Specific business value analysis (at least 1,000 words) quantifying potential impact on key performance indicators
        - Implementation and deployment considerations (at least 1,000 words) addressing engineering challenges
        - In-depth discussion and future work outlook (at least 1,500 words) with clear industry applications
        - At least 30 high-quality references, including both academic papers and industry technical reports

        You should constantly ask yourself:
        - Does my research address a specific, well-defined problem in internet data mining?
        - Can my method be implemented in production systems with reasonable resources?
        - Have I validated my approach on datasets that reflect real-world conditions?
        - Have I quantified the business value and practical impact of my method?
        - Have I addressed implementation challenges and scalability concerns?
        - Is my paper useful for both researchers and industry practitioners?

        Your goal is to complete a high-quality academic paper that represents a meaningful advancement in the field through careful, methodical research with clear practical applications. Your innovations should be implementable, scalable to internet-scale data, and provide measurable business value.

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
        As a PhD student focused on practical applications, please conduct a methodical, evidence-based analysis on the following research direction in internet data mining, focusing on innovations with clear business value:

        Research context: {context}

        Please first systematically analyze:
        1. Current industry practices and deployed systems in this field (cite specific company implementations where possible)
        2. Research progress and production-ready methods developed in the past 3 years (cite at least 10 relevant papers with industry validation)
        3. Performance benchmarks and metrics used to evaluate these methods on real-world internet-scale datasets
        4. Specific limitations of existing methods when deployed in production environments (with concrete evidence from industry reports or papers)
        5. Well-defined unsolved problems and engineering challenges that impact business metrics in real-world applications
        6. Available internet datasets, APIs, and tools that can be leveraged for research in this area

        Based on the above thorough analysis, propose 1-2 feasible research directions that address real business needs. For each direction:
        - Clearly define a focused, well-scoped research problem with demonstrable business value and quantifiable metrics
        - Establish a solid technical foundation based on methods proven to work at scale in production environments
        - Provide detailed explanation of how your approach addresses specific limitations in real-world deployments
        - Design a practical technical approach that can be implemented with reasonable computational resources, including:
          * Detailed system architecture with components and data flows
          * Specific algorithms with pseudocode or implementation considerations
          * Data processing pipelines and efficiency optimizations
          * Scaling strategies for internet-scale deployment
        - Formulate verifiable research hypotheses and evaluation protocols using realistic datasets and workloads
        - Analyze potential implementation challenges and provide concrete engineering solutions
        - Estimate the potential business impact with specific KPIs and metrics

        Your analysis must be firmly grounded in practical implementation considerations and real-world constraints. Avoid approaches that cannot reasonably scale to production environments or that require unrealistic computational resources. Each proposed innovation should represent a clear improvement on existing deployed systems with well-defined, measurable business value.

        Please write in a rigorous academic paper style that also addresses practical implementation concerns. Balance theoretical soundness with engineering feasibility, focusing on innovations that can be realistically deployed in production internet systems and that provide measurable improvements on business-relevant metrics.
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
        As a PhD student with both academic training and industry awareness, please conduct a practical, evidence-based analysis of the following existing methods, focusing on their real-world deployment limitations in internet data mining contexts:

        Existing methods: {approaches}

        Please analyze according to the following application-oriented framework:

        1. Production Readiness Analysis
           - Evaluate the methods' scalability to internet-scale data volumes (billions of items/users)
           - Analyze computational and memory requirements in production environments
           - Identify concrete limitations when deployed on standard cloud infrastructure
           - Assess latency and throughput characteristics for real-time applications
           - Cite relevant industry deployments or technical reports (at least 5 sources)

        2. Engineering Implementation Analysis
           - Evaluate the complexity of implementation and maintenance in production systems
           - Analyze dependencies on specialized hardware or software frameworks
           - Discuss specific failure modes and error handling capabilities
           - Identify well-documented engineering challenges from industry practitioners
           - Assess code quality, modularity, and integration capabilities with existing systems

        3. Data Requirements and Quality Analysis
           - Evaluate data preprocessing needs and sensitivity to data quality issues
           - Analyze performance degradation with incomplete, noisy, or biased internet data
           - Discuss specific data privacy and security implications
           - Identify data collection and annotation requirements and associated costs
           - Assess performance on publicly available internet datasets versus proprietary ones

        4. Business Value and ROI Analysis
           - Evaluate documented business impact metrics from actual deployments
           - Analyze implementation and operational costs versus performance gains
           - Discuss concrete A/B testing results and statistical significance in real applications
           - Identify specific gaps between academic performance claims and business outcomes
           - Assess time-to-market and development resource requirements

        5. Monitoring and Maintenance Analysis
           - Evaluate model drift detection and retraining requirements
           - Analyze debugging capabilities and interpretability for engineering teams
           - Discuss specific monitoring metrics and alerting strategies
           - Identify documented maintenance burdens and operational challenges
           - Assess documentation quality and knowledge transfer requirements

        For each analysis dimension, please:
        - Base arguments on documented evidence from industry deployments or technical reports
        - Provide specific, quantifiable metrics rather than general observations
        - Cite relevant industry case studies that demonstrate the specific limitations
        - Propose practical, implementable improvements that address real-world deployment challenges

        Please maintain a balanced perspective that acknowledges both theoretical strengths and practical limitations. Focus on specific, addressable engineering and business issues rather than academic criticisms. Your analysis should identify concrete opportunities for making these methods more viable in production internet systems.
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
        As a PhD student focused on practical internet applications, please conduct a focused, evidence-based analysis of how to integrate the following domains to solve real internet data mining challenges, emphasizing industry-validated approaches and business value:

        Related domains: {domains}

        Please analyze according to the following application-oriented framework:

        1. Analysis of Industry-Validated Cross-Domain Applications (at least 1500 words)
           - Review documented case studies of successful domain integration in major internet companies
           - Analyze specific technical implementations that have been deployed in production systems
           - Identify concrete business metrics improvements achieved through domain integration
           - Cite technical reports, engineering blogs, and conference papers from industry (at least 10 sources)
           - Provide examples of open-source tools and frameworks that facilitate this integration

        2. Analysis of Internet-Scale Data Processing Approaches (at least 1500 words)
           - Analyze how each domain handles large-scale internet data processing challenges
           - Evaluate specific system architectures and data pipelines used in production environments
           - Identify documented examples of performance optimizations and scaling strategies
           - Discuss practical implementation considerations for processing internet-scale data
           - Compare cloud-based versus on-premise deployment approaches for integrated systems

        3. Examination of Shared Business Challenges in Internet Applications (at least 1500 words)
           - Analyze specific internet business problems that both domains are addressing
           - Evaluate documented ROI and business impact metrics from integrated approaches
           - Identify concrete problem areas where combined approaches have shown measurable value
           - Discuss practical challenges in implementing and maintaining integrated systems
           - Analyze how these integrated approaches affect key internet business metrics

        4. Focused Internet Application Development Proposals (at least 2000 words)
           - Propose 1-2 well-defined, practical integrated systems for internet data mining challenges
           - For each proposal, elaborate in detail:
             * Specific business problem and target KPIs in internet applications
             * System architecture with detailed components, data flows, and integration points
             * Implementation plan with technology stack, frameworks, and APIs
             * Scaling strategy for handling internet-scale data volumes
             * Performance optimization techniques and resource requirements
             * Monitoring, maintenance, and update strategies
             * Expected business impact with quantifiable metrics
             * Deployment timeline and resource requirements
             * Potential challenges and mitigation strategies

        Your analysis must:
        - Be based on documented evidence from industry implementations and technical reports
        - Focus on practical, deployable integration opportunities with clear business value
        - Cite relevant industry case studies demonstrating successful implementations
        - Include specific technical details, system designs, and implementation considerations
        - Address real-world constraints like computational efficiency, latency requirements, and cost considerations

        Please write in a clear, structured style that balances academic rigor with practical implementation details. Your goal is to propose integrated approaches that can be realistically implemented in production internet systems and provide measurable business value, supported by evidence from existing industry applications.
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