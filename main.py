import os
from agents.phd_student import PhDStudentAgent
from agents.academic_advisor import AcademicAdvisorAgent
from agents.industry_advisor import IndustryAdvisorAgent
from modules.knowledge_retrieval import KnowledgeRetrievalModule
from modules.paper_evaluation import PaperEvaluationModule
from coordinator import AgentCoordinator

# 设置API密钥
os.environ["DEEPSEEK_API_KEY"] = "sk-f97b662a15ad42318579bb9b21a80db1"

def main():
    # 初始化各个Agent
    phd_student = PhDStudentAgent()
    academic_advisor = AcademicAdvisorAgent()
    industry_advisor = IndustryAdvisorAgent()

    # 初始化模块
    knowledge_module = KnowledgeRetrievalModule()
    evaluation_module = PaperEvaluationModule()

    # 初始化协调器
    coordinator = AgentCoordinator(
        phd_student=phd_student,
        academic_advisor=academic_advisor,
        industry_advisor=industry_advisor,
        knowledge_module=knowledge_module,
        evaluation_module=evaluation_module
    )

    # 运行系统
    coordinator.start_interaction()

    # 保存交互历史和论文草稿
    coordinator.save_interaction_history("interaction_history.json")
    coordinator.save_paper_drafts("paper_drafts.json")

    # 保存代理记忆
    coordinator.save_agent_memories("memories")

    # 保存最终论文
    save_final_paper(coordinator.phd_student.paper_draft)

    print("\n论文已保存到 final_paper.md 文件中，可以直接查看。")
    print("代理记忆已保存到 memories 目录中。")

def save_final_paper(paper_content: str) -> None:
    """
    保存最终论文到文件

    Args:
        paper_content: 论文内容
    """
    with open("final_paper.md", "w", encoding="utf-8") as f:
        f.write(paper_content)

if __name__ == "__main__":
    main()