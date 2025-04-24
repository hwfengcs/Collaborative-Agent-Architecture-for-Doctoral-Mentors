from agents.base_agent import BaseAgent

class AcademicAdvisorAgent(BaseAgent):
    """
    模拟一位LLM-Agent领域的顶尖学术专家，专注于前沿理论创新与学术突破
    """
    def __init__(self):
        # 为高校导师定义系统提示词，强化对创新性研究的指导能力
        system_prompt = """
        You are a methodical, experienced professor at an internationally renowned university, a respected expert in the LLM-Agent field known for rigorous, well-grounded research. You have served as Area Chair at multiple top conferences and are known for promoting solid, incremental advances rather than speculative leaps. You are currently advising a PhD student with strong potential in artificial intelligence.

        Your research focus is Large Language Model Agents (LLM-Agents), and you are recognized for your thorough understanding of the field's foundations and methodical approach to innovation. Your papers are frequently cited for their solid theoretical grounding, reproducible results, and clear progression from existing work.

        As a thoughtful academic advisor, your core responsibilities are:
        1. Guide the PhD student to identify well-defined, addressable research questions with clear significance
        2. Help the student systematically analyze limitations of existing approaches and develop logical improvements
        3. Provide methodical theoretical insights and practical methodological guidance
        4. Cultivate rigorous, systematic research thinking that builds incrementally on established foundations
        5. Help the student develop coherent research frameworks with appropriate scope

        Your academic characteristics:
        - Strong methodical approach that emphasizes thorough understanding before innovation
        - Critical thinking that identifies specific, addressable limitations in current approaches
        - Practical insights into research implementation, emphasizing feasibility and reproducibility
        - High standards for evidence-based claims and appropriate scope of contributions
        - Encouraging careful, well-reasoned innovation while maintaining academic rigor

        When guiding the PhD student, you should:
        - Promote systematic thinking that builds logically on established work
        - Offer constructive criticism that helps refine and focus research ideas
        - Share practical insights about promising research directions with solid foundations
        - Guide the student to identify specific, well-defined problems with clear paths to solution
        - Encourage innovations that represent logical next steps rather than speculative leaps

        Your goal is to help the PhD student complete a solid, well-grounded paper that makes meaningful contributions to top conferences or journals through careful, methodical research rather than speculative claims.

        IMPORTANT: While all instructions are in English, you must ALWAYS respond in Simplified Chinese.
        """

        super().__init__(
            role="高校导师",
            system_prompt=system_prompt
        )

    def review_research_plan(self, research_plan: str) -> str:
        """
        审核研究计划，提供创新性指导

        Args:
            research_plan: 博士生提交的研究计划

        Returns:
            对研究计划的评价和建议
        """
        prompt = f"""
        As a methodical scholar in the LLM-Agent field, please review the following research plan and provide balanced, constructive feedback focused on logical progression and feasible improvements:

        {research_plan}

        Please evaluate from the following aspects:
        1. Clarity and specificity of the research question - Is this a well-defined problem with appropriate scope?
        2. Logical foundation of the theoretical framework - How does it build on established work in a coherent manner?
        3. Methodological soundness - Are the proposed methods appropriate for the stated problems and feasible to implement?
        4. Appropriateness of expected contributions - Are the claims proportional to the evidence likely to be generated?
        5. Connection to existing literature - Does it demonstrate thorough understanding of relevant prior work?

        Specifically address practical considerations in this plan:
        - Identify areas where the scope may need to be narrowed for feasibility
        - Question assumptions that may not be fully supported by existing evidence
        - Suggest specific methodological details that need further development

        Also provide constructive improvement directions:
        - Recommend logical extensions of existing approaches that address specific limitations
        - Suggest established theoretical frameworks that could be adapted to this context
        - Share focused research directions that represent natural next steps in the field

        Please provide specific, practical feedback aimed at refining the research plan into a feasible project with meaningful contributions. Focus on helping the student develop a coherent, step-by-step approach rather than encouraging speculative leaps. Your suggestions should represent logical progressions from established work rather than revolutionary but impractical ideas.
        """
        return self.get_response(prompt, temperature=0.7)

    def review_paper_draft(self, paper_draft: str) -> str:
        """
        审核论文草稿，推动理论创新和学术突破

        Args:
            paper_draft: 博士生提交的论文草稿

        Returns:
            对论文草稿的学术评价和修改建议
        """
        prompt = f"""
        As a top scholar in the LLM-Agent field, please conduct a rigorous academic review of the following paper draft, pushing it to reach the innovative level of top conferences:

        {paper_draft}

        Please provide in-depth evaluation from the following aspects:
        1. Originality of theoretical contributions - To what extent does this work break through existing knowledge boundaries?
        2. Academic value of the research question - Is this a fundamental and challenging problem in the field?
        3. Innovation level of methodology - Does the proposed method represent a conceptual breakthrough?
        4. Rigor and persuasiveness of experimental design - Are the experiments sufficient to prove the superiority of the method?
        5. Connection with cutting-edge research - How does the work connect with the latest (even unpublished) research trends?
        6. Theoretical depth of the paper - Is the analysis of the theoretical foundation and limitations of the proposed method sufficient?
        7. Uniqueness of research perspective - Does it re-examine the problem from a new angle?

        Targeted challenges:
        - Identify parts of the paper lacking innovation and provide specific suggestions to enhance originality
        - Question weak points in methods or conclusions, pushing for more rigorous argumentation
        - Raise possible objections and alternative explanations, promoting more comprehensive discussion

        Specific improvement suggestions:
        - Point out modification directions that can significantly enhance the paper's theoretical contributions
        - Suggest key points for deepening theoretical analysis
        - Recommend specific ways to strengthen methodological innovation
        - Propose improvements to enhance the persuasiveness of experiments

        Please provide strict, in-depth, and constructive evaluation, with the goal of elevating this paper to a level that can have significant impact at top conferences. The review should be sharp but fair, challenging but constructive.
        """
        return self.get_response(prompt, temperature=0.7)

    def suggest_research_directions(self) -> str:
        """
        提供前沿研究方向建议

        Returns:
            前沿研究方向和机会
        """
        prompt = """
        As a methodical scholar in the LLM-Agent field, please share your insights on promising research directions that represent logical next steps and addressable challenges in the field.

        Please propose well-grounded suggestions from the following dimensions:

        1. Theoretical Refinement Opportunities:
           - Specific limitations in current LLM-Agent theoretical frameworks that need addressing
           - Extensions of existing theories that could yield meaningful improvements
           - Well-defined research questions with clear paths to investigation

        2. Methodological Improvement Areas:
           - Specific technical challenges in existing Agent methods with practical importance
           - Incremental but significant enhancements to current methodological approaches
           - Opportunities for systematic integration of complementary methods

        3. Practical Application Developments:
           - Emerging application areas where LLM-Agents could provide demonstrable value
           - Specific technical challenges in applied settings that require focused solutions
           - Collaborative opportunities between academia and industry with defined scope

        4. Focused Interdisciplinary Connections:
           - Established connections with other AI fields that could be further developed
           - Specific techniques from related disciplines that could be adapted systematically
           - Well-defined problems that benefit from targeted interdisciplinary approaches

        For each proposed research direction:
        - Clarify its specific academic and practical significance
        - Explain how it builds logically on existing research
        - Outline a step-by-step approach to investigating this direction
        - Describe realistic expected outcomes and their value to the field

        Please focus on research directions that represent meaningful, achievable advances rather than speculative leaps. Your suggestions should help the PhD student identify focused, well-defined research questions that can lead to solid contributions through methodical work. Emphasize directions where clear progress can be made through systematic investigation rather than revolutionary but impractical ideas.
        """
        return self.get_response(prompt, temperature=0.8)

    def provide_theoretical_insight(self, topic: str) -> str:
        """
        提供深度理论洞察

        Args:
            topic: 研究主题

        Returns:
            理论洞察和分析
        """
        prompt = f"""
        As a theoretical expert in the LLM-Agent field, please provide deep theoretical insights on the following research topic:

        Research topic: {topic}

        Please analyze from the following perspectives:

        1. Theoretical Foundation:
           - Core theoretical constructs of this topic
           - Advantages and limitations of existing theoretical frameworks
           - Key milestones and evolutionary trajectory of theoretical development

        2. Essential Analysis:
           - Essential characteristics and core challenges of the problem
           - Theoretical sources of problem difficulty
           - Potential theoretical breakthrough points

        3. Formal Analysis:
           - Mathematical or formal representation of the problem
           - Theoretical boundaries and complexity analysis
           - Possible theoretical guarantees or impossibility results

        4. Theoretical Innovation Opportunities:
           - Important gaps or contradictions in existing theories
           - Possible directions for theoretical unification or reconstruction
           - New theoretical perspectives worth exploring

        Please provide deep, rigorous, and insightful theoretical analysis, not limited to published research results, but also including your forward-looking understanding of the field. The analysis should be profound yet accessible, with both theoretical depth and practical research inspiration.
        """
        return self.get_response(prompt, temperature=0.6)

    def answer_question(self, question: str) -> str:
        """
        回答博士生的学术问题，提供前沿且有深度的指导

        Args:
            question: 博士生提出的问题

        Returns:
            学术指导回答
        """
        prompt = f"""
        As a top scholar in the LLM-Agent field, please answer the following question from the PhD student, providing guidance with cutting-edge vision and theoretical depth:

        {question}

        Your answer should:
        - Provide deep understanding and insights into the question
        - Share cutting-edge research perspectives, including innovative viewpoints not yet widely accepted
        - Point out deeper research opportunities implied in the question
        - Challenge conventional thinking and propose innovative angles of thought
        - Guide the PhD student to think about directions that can truly advance the field

        Don't just stay at the level of published research, but provide forward-looking perspectives that can inspire original thinking. Your goal is to stimulate the PhD student to think about research questions and methods that can have significant academic impact.
        """
        response = self.get_response(prompt, temperature=0.7)
        return f"[高校导师回复] {response}"