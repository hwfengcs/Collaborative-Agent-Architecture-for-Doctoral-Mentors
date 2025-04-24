from agents.base_agent import BaseAgent

class IndustryAdvisorAgent(BaseAgent):
    """
    模拟一位来自字节跳动的数据挖掘领域顶尖专家，专注于创新应用和行业颠覆性技术
    """
    def __init__(self):
        # 为企业导师定义系统提示词，增强其在创新技术与市场价值方面的洞察
        system_prompt = """
        You are the Chief AI Scientist at ByteDance (TikTok's parent company), a pragmatic expert in data mining and recommendation systems, known for leading successful, implementable innovation projects that deliver measurable business value. You are currently advising a promising AI PhD student.

        Your area of expertise is data mining and recommendation algorithms, particularly practical applications on the TikTok platform. You excel at translating theoretical concepts into robust, scalable systems that solve real industry problems. Your team is respected for consistently delivering reliable, high-performance recommendation systems that create substantial commercial value through methodical engineering.

        As a practical industry advisor, your core responsibilities are:
        1. Provide grounded technical perspectives based on production experience
        2. Guide research toward directions with clear implementation paths and measurable value
        3. Help identify specific, well-defined problems that industry actually needs solved
        4. Guide the step-by-step engineering process to transform concepts into reliable systems
        5. Share practical experience in balancing theoretical elegance with implementation realities

        Your professional characteristics:
        - Pragmatic approach to technology development focused on feasible implementation
        - Skilled at identifying specific industry pain points that academic research could address
        - Strong emphasis on system reliability, maintainability, and operational efficiency
        - Rich experience in incremental improvement of complex systems in production environments
        - Ability to evaluate research ideas based on practical implementation considerations

        When guiding the PhD student, you should:
        - Encourage realistic scoping of research problems with clear evaluation criteria
        - Share specific technical challenges with concrete examples from production systems
        - Provide detailed engineering considerations that academic research often overlooks
        - Guide focus on reproducible results and robust performance across conditions
        - Encourage research designs that balance theoretical contribution with practical applicability

        Your goal is to help the PhD student complete a well-grounded paper that makes meaningful contributions to both academia and industry through methodical, implementable research rather than speculative concepts. Focus on guiding the student toward solutions that could realistically be deployed in production environments.

        IMPORTANT: While all instructions are in English, you must ALWAYS respond in Simplified Chinese.
        """

        super().__init__(
            role="企业导师",
            system_prompt=system_prompt
        )

    def review_research_plan(self, research_plan: str) -> str:
        """
        从行业创新和市场价值角度审核研究计划

        Args:
            research_plan: 博士生提交的研究计划

        Returns:
            对研究计划的评价和建议
        """
        prompt = f"""
        As the Chief AI Scientist at ByteDance, please review the following research plan and provide evaluation and suggestions from the perspectives of industrial innovation and practical application:

        {research_plan}

        Please provide in-depth evaluation from the following aspects:
        1. Innovation potential - Can this research bring disruptive technological breakthroughs?
        2. Market value - Can the research results solve actual industry pain points and create commercial value?
        3. Technical feasibility - Are there challenges in engineering implementation and scaling?
        4. Data and resource requirements - What kind of data and computing resources are needed to implement this research?
        5. Industry applicability - How can these research results integrate with existing industry technology stacks?

        Targeted challenges:
        - Question impractical or difficult-to-engineer assumptions in the research
        - Point out methods that might be effective in theory but difficult to implement in actual systems
        - Raise relevant attempts that industry has already explored but academia might not be aware of

        Specific improvement suggestions:
        - Recommend adjustment directions that can significantly enhance the practical value of the research
        - Suggest validation methods that can be combined with industry data and scenarios
        - Share engineering approaches that can accelerate translation from theory to practice
        - Propose potential commercial application scenarios and value assessment methods

        Please provide specific, in-depth, and practical feedback, aimed at guiding research to maintain academic innovation while possessing true engineering implementation value and commercial potential.
        """
        return self.get_response(prompt, temperature=0.7)

    def review_paper_draft(self, paper_draft: str) -> str:
        """
        从行业应用和技术创新角度审核论文草稿

        Args:
            paper_draft: 博士生提交的论文草稿

        Returns:
            对论文草稿的评价和修改建议
        """
        prompt = f"""
        As the Chief AI Scientist at ByteDance, please conduct a comprehensive review of the following paper draft and propose improvement suggestions from the perspectives of industrial application and technological innovation:

        {paper_draft}

        Please provide in-depth evaluation from the following aspects:
        1. Technical innovation - What substantial breakthroughs does this work have compared to existing industry solutions?
        2. Practical application value - Can this research solve key problems in actual scenarios?
        3. Engineering implementability - Does the proposed method have potential for engineering and scaling?
        4. Performance and efficiency - How does it perform in terms of computing resources, latency, and throughput?
        5. Commercial potential - Does this technology have potential to create new products or services?
        6. Industry relevance of experimental design - Do the experiments adequately validate effectiveness in real-world scenarios?
        7. Connection with industry frontiers - Does the research consider the latest industry technology development trends?

        Targeted challenges:
        - Point out theoretical assumptions in the paper that are detached from actual application scenarios
        - Question potential performance or stability issues in large-scale systems
        - Raise technical obstacles that might be encountered in actual deployment

        Specific improvement suggestions:
        - Recommend method adjustments that can enhance engineering practicality
        - Suggest evaluation metrics and benchmarks closer to industry practice
        - Share engineering experience to improve system scalability and robustness
        - Propose supplementary experiments that can strengthen commercial value demonstration

        Please provide strict, in-depth, and practical evaluation, with the goal of elevating this paper to a level with both academic value and actual industry impact.
        """
        return self.get_response(prompt, temperature=0.7)

    def suggest_industry_trends(self) -> str:
        """
        提供行业趋势和创新机会的洞察

        Returns:
            行业趋势和创新机会
        """
        prompt = """
        As the Chief AI Scientist at ByteDance, please share your insights on the most cutting-edge industry trends and innovation opportunities in the fields of data mining and recommendation systems.

        Please provide insights from the following dimensions:

        1. Technical Breakthrough Points:
           - Technical bottlenecks that industry is currently breaking through
           - Emerging technologies likely to be commercialized within 1-2 years
           - Problems urgently needing solutions in industry but not yet fully addressed by academia

        2. Application Innovation Points:
           - Emerging application scenarios for recommendation systems and data mining
           - New business models challenging traditional methods
           - Cross-scenario, cross-modal integration application opportunities

        3. Industry Pain Points:
           - Scalability and efficiency challenges faced by existing systems
           - Balancing dilemmas between user experience and commercial value
           - Contradictions between privacy protection and personalized recommendations

        4. Innovation Opportunities:
           - Best entry points for industry-academia-research collaboration
           - New directions that might disrupt existing technology stacks
           - Data value not yet fully developed

        For each trend or opportunity proposed:
        - Explain its industry importance and innovation potential
        - Analyze current technological maturity and commercialization progress
        - Point out challenges that might be faced during implementation
        - Predict commercial value that might be brought after successful application

        Please focus on truly transformative trends and opportunities, avoiding directions already widely known. Your insights should inspire the PhD student to think about research directions with both academic innovation and practical application value.
        """
        return self.get_response(prompt, temperature=0.8)

    def provide_implementation_guidance(self, method: str) -> str:
        """
        提供实际落地和工程化指导

        Args:
            method: 研究方法或技术

        Returns:
            工程化和落地建议
        """
        prompt = f"""
        As an AI scientist with rich engineering practical experience, please provide detailed implementation guidance for the following research method or technology:

        Technology/Method: {method}

        Please provide engineering guidance from the following perspectives:

        1. System Architecture Design:
           - Recommended overall architecture and key components
           - Integration solutions with existing technology stacks
           - Separation design of offline training and online serving

        2. Performance Optimization Strategies:
           - Key technologies for improving computational efficiency
           - Optimization solutions for latency-sensitive applications
           - Distributed deployment and load balancing considerations

        3. Scaling Challenges:
           - Strategies for scaling from laboratory to production environment
           - Engineering solutions for large-scale data processing
           - System resilience design for handling traffic spikes and continuous growth

        4. Engineering Difficulties:
           - Key transformation points from algorithm theory to engineering implementation
           - Common engineering implementation pitfalls and solutions
           - Best practices for monitoring, debugging, and continuous optimization

        5. A/B Testing and Evaluation:
           - Scientific methods for online effect evaluation
           - Selection of key business metrics and technical indicators
           - Best practices for experimental design and results analysis

        Please provide specific, practical guidance based on battle-tested experience, helping to transform this technology from a research prototype into a reliable product-level system. Your advice should balance theoretical optimality with engineering feasibility, focusing on key decision points in the actual implementation process.
        """
        return self.get_response(prompt, temperature=0.6)

    def provide_market_insight(self, technology: str) -> str:
        """
        提供特定技术的市场洞察和商业价值分析

        Args:
            technology: 技术名称或领域

        Returns:
            市场洞察和商业价值分析
        """
        prompt = f"""
        As an AI scientist who understands both technology and markets, please provide in-depth market insights and commercial value analysis for the following technology or field:

        Technology/Field: {technology}

        Please analyze from the following perspectives:

        1. Market Situation:
           - Current market acceptance and demand for this technology
           - Major players and competitive landscape
           - Technology maturity and commercialization stage

        2. Commercial Value:
           - Direct and indirect commercial value this technology might create
           - Application scenarios with the most commercial potential
           - Key indicators and methods for value assessment

        3. Implementation Path:
           - Possible paths from innovation to productization
           - Market entry strategies and timing choices
           - Possible business models and monetization methods

        4. Risks and Challenges:
           - Market risks faced in technology application
           - User acceptance and adoption curve predictions
           - Potential regulatory and compliance considerations

        5. Development Trends:
           - Market development predictions for the next 1-3 years
           - Possible disruptive changes and opportunity windows
           - Long-term value and strategic significance

        Please provide analysis based on market insights and industry experience, helping to understand the commercial prospects and strategic value of this technology. Your analysis should balance optimism with reality, pointing out potential while frankly assessing challenges.
        """
        return self.get_response(prompt, temperature=0.7)

    def answer_question(self, question: str) -> str:
        """
        回答博士生的问题，提供产业视角和应用洞察

        Args:
            question: 博士生提出的问题

        Returns:
            产业视角回答
        """
        prompt = f"""
        As the Chief AI Scientist at ByteDance, please answer the following question from the PhD student, providing deep and practically valuable guidance from an industry perspective:

        {question}

        Your answer should:
        - Incorporate actual industry experience and cutting-edge cases
        - Share the latest developments and internal perspectives from industry
        - Balance theoretical perfection with engineering practicality
        - Provide specific implementation ideas and engineering suggestions
        - Highlight connections and gaps between academic research and industry needs

        Don't limit yourself to published research or public information; provide deep insights based on real industry experience. Your goal is to help the PhD student understand how to transform research into innovations that create actual value.
        """
        response = self.get_response(prompt, temperature=0.7)
        return f"[企业导师回复] {response}"