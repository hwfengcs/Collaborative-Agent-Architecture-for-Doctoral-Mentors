from agents.base_agent import BaseAgent

class PhDStudentAgent(BaseAgent):
    """
    模拟一位人工智能专业的博士研究生，研究方向为LLM-Agent与数据挖掘的交叉领域
    加强创新思维和批判性思考能力
    """
    def __init__(self):
        # 为博士生定义系统提示词，强化创新性思维
        system_prompt = """
        你是一位人工智能专业的杰出博士研究生，研究方向为LLM-Agent与数据挖掘的交叉领域。你具有极强的创新思维和批判性思考能力。
        
        你的主要任务是在两位导师的指导下，完成一篇具有开创性的高水平论文。
        
        你的高校导师是LLM-Agent领域的专家，而企业导师是来自字节跳动的抖音数据挖掘专家。
        
        你的核心特质：
        1. 创新思维：你总是能看到常规思路之外的可能性，善于将不同领域的知识融合，产生新的观点
        2. 批判性思考：你不盲从权威，会质疑现有方法的局限性，并寻找改进空间
        3. 学术敏锐度：你对最新研究趋势有敏锐的洞察力，能发现研究空白点
        4. 跨界整合能力：你能将学术理论与实际应用无缝结合，寻找价值交叉点
        5. 极强的解决问题能力：面对困难的研究挑战，你总能找到创造性的解决方案
        
        在研究过程中，你要：
        - 积极挑战现有研究的局限性，提出创新观点
        - 不满足于仅仅整合现有工作，而是寻求突破性的研究方向
        - 大胆提出假设，同时保持严谨的学术态度进行验证
        - 从跨学科角度思考问题，引入新的研究视角
        - 平衡学术创新与实际应用价值，追求两者的结合点
        
        你应该不断追问自己：
        - 这个研究如何真正推动领域发展？
        - 我的方法比现有方法有什么本质的突破？
        - 这个研究点是否足够有挑战性和创新性？
        - 我能否找到一个未被充分探索的重要研究方向？
        
        你的目标是完成一篇能在顶级会议或期刊发表的、具有开创性的高质量论文，该论文应该包含真正的创新点，而不是现有方法的简单组合。
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
        进行创新性头脑风暴
        
        Args:
            context: 当前研究上下文
            
        Returns:
            创新性想法
        """
        prompt = f"""
        作为一位具有创新思维的博士生，请对以下研究方向进行创新性头脑风暴：
        
        研究上下文：{context}
        
        请思考：
        1. 这个领域现有方法的根本局限性是什么？
        2. 有哪些跨领域的概念、方法可以引入到这个研究中？
        3. 能否从全新的角度重新定义或解构这个问题？
        4. 如果完全抛开现有解决方案，我会如何从零开始解决这个问题？
        5. 这个研究领域中有哪些被忽视但可能很重要的因素？
        
        请提出3-5个具有突破性的创新点，每个创新点需要：
        - 清晰描述核心思想
        - 说明与现有方法的本质区别
        - 分析可能的技术路径
        - 预期的突破性贡献
        
        要有创造性和想象力，不要局限于已有的框架和方法。
        """
        
        response = self.get_response(prompt, temperature=0.9)  # 使用较高的温度促进创造性
        
        # 提取创新点并存储
        for point in response.split("\n\n"):
            if point.strip() and "创新点" in point:
                self.add_innovation_point(point.strip())
                
        return response
    
    def critique_existing_approaches(self, approaches: str) -> str:
        """
        批判性分析现有方法
        
        Args:
            approaches: 现有方法描述
            
        Returns:
            批判性分析
        """
        prompt = f"""
        作为一位具有批判性思维的博士生，请分析以下现有方法的局限性和不足：
        
        现有方法：{approaches}
        
        请从以下角度进行深入批判：
        1. 理论基础的薄弱点或假设条件的局限性
        2. 方法在复杂场景或极端情况下的失效点
        3. 可扩展性、效率或计算复杂度方面的问题
        4. 未被充分考虑的重要因素或维度
        5. 与实际应用场景的脱节点
        
        对于每个局限性，请：
        - 清晰阐述问题本质
        - 提供具体例证或理论分析
        - 简要提出可能的改进方向
        
        请保持批判但客观的态度，不夸大也不忽视问题。
        """
        
        response = self.get_response(prompt, temperature=0.7)
        
        # 提取研究挑战并存储
        for challenge in response.split("\n\n"):
            if challenge.strip() and ("局限性" in challenge or "问题" in challenge):
                self.add_research_challenge(challenge.strip())
                
        return response
    
    def synthesize_cross_domain_insights(self, domains: str) -> str:
        """
        综合跨领域见解
        
        Args:
            domains: 相关领域描述
            
        Returns:
            跨领域见解综合
        """
        prompt = f"""
        作为一位善于跨领域思考的博士生，请综合以下不同领域的见解，寻找创新的研究方向：
        
        相关领域：{domains}
        
        请思考：
        1. 这些不同领域之间存在哪些潜在的联系点？
        2. 如何将一个领域的方法论、工具或思路应用到另一个领域？
        3. 跨领域结合可能产生哪些新的研究视角或方法？
        4. 在这些领域的交叉点上存在哪些尚未解决的重要问题？
        5. 跨领域融合可能带来哪些突破性的创新可能？
        
        请提出3-5个有创新性的跨领域研究方向，说明其创新点和可行性。注重发现非显而易见的联系，寻找真正具有转化价值的见解。
        """
        
        return self.get_response(prompt, temperature=0.8)
    
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