from openai import OpenAI
import os
import datetime
from typing import List, Dict, Any, Optional

class BaseAgent:
    """
    所有Agent的基类，提供基本的对话和记忆功能
    """
    def __init__(
        self,
        role: str,
        system_prompt: str,
        model: str = "deepseek-r1"
    ):
        """
        初始化Agent

        Args:
            role: Agent的角色名称
            system_prompt: 系统提示词，定义Agent的行为和知识
            model: 使用的模型名称
        """
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.conversation_history = []
        self.memory_bank = []  # 长期记忆存储

        # 添加系统提示到对话历史
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})

    def get_response(self, message: str, temperature: float = 0.7) -> str:
        """
        获取模型回复

        Args:
            message: 输入的消息
            temperature: 温度参数，控制响应的随机性

        Returns:
            模型的回复
        """
        # 添加用户消息到历史
        self.conversation_history.append({"role": "user", "content": message})

        # 调用API获取回复
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature
            )
            reply = response.choices[0].message.content

            # 添加模型回复到历史
            self.conversation_history.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            error_message = f"API调用出错: {str(e)}"
            print(error_message)
            return error_message

    def add_message_to_history(self, role: str, content: str) -> None:
        """
        手动添加消息到对话历史

        Args:
            role: 消息的角色 (system, user, assistant)
            content: 消息内容
        """
        self.conversation_history.append({"role": role, "content": content})

    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        清除对话历史

        Args:
            keep_system_prompt: 是否保留系统提示词
        """
        if keep_system_prompt and self.system_prompt:
            self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史

        Returns:
            对话历史列表
        """
        return self.conversation_history

    def create_memory(self, phase: str, content: str) -> None:
        """
        创建新的长期记忆

        Args:
            phase: 记忆所属的阶段
            content: 记忆内容
        """
        memory = {
            "phase": phase,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.memory_bank.append(memory)

    def summarize_phase(self, phase: str, context: Optional[str] = None) -> str:
        """
        总结特定阶段的关键内容，并将其存储为长期记忆

        Args:
            phase: 要总结的阶段名称
            context: 可选的额外上下文信息

        Returns:
            生成的总结内容
        """
        # 构建提示词，要求模型总结阶段内容
        prompt = f"""
        请总结"{phase}"阶段的关键内容和重要发现。

        {f"上下文信息：{context}" if context else ""}

        请生成一个简洁但全面的总结（500-800字），包含：
        1. 该阶段的主要目标和完成情况
        2. 关键决策和重要发现
        3. 达成的共识和结论
        4. 对下一阶段的影响和启示

        总结应该保留所有关键信息，以便在未来阶段中可以作为参考，而不需要回顾原始对话。
        """

        # 获取总结
        summary = self.get_response(prompt, temperature=0.5)

        # 存储为长期记忆
        self.create_memory(phase, summary)

        return summary

    def get_memories(self, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取特定阶段或所有的长期记忆

        Args:
            phase: 可选的阶段名称过滤器

        Returns:
            记忆列表
        """
        if phase:
            return [memory for memory in self.memory_bank if memory["phase"] == phase]
        return self.memory_bank

    def inject_memories_to_context(self, phases: Optional[List[str]] = None) -> None:
        """
        将长期记忆注入到当前对话上下文中

        Args:
            phases: 可选的要注入的阶段列表，如果为None则注入所有记忆
        """
        memories_to_inject = []

        if phases:
            for phase in phases:
                phase_memories = self.get_memories(phase)
                memories_to_inject.extend(phase_memories)
        else:
            memories_to_inject = self.memory_bank

        if not memories_to_inject:
            return

        # 构建记忆内容
        memory_content = "以下是之前阶段的关键总结，请参考这些信息：\n\n"

        for memory in memories_to_inject:
            memory_content += f"--- {memory['phase']} 阶段总结 ---\n{memory['content']}\n\n"

        # 将记忆作为系统消息注入到对话历史中
        self.add_message_to_history("system", memory_content)