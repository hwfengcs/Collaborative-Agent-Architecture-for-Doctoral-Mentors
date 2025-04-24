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
        model: str = "deepseek-reasoner"
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

    def manage_history_length(self, max_tokens: int = 60000) -> None:
        """
        管理对话历史长度，避免超过模型的最大上下文长度限制

        Args:
            max_tokens: 最大允许的token数量
        """
        # 简单估算token数量（实际应用中可能需要更精确的计算方法）
        estimated_tokens = sum(len(msg["content"]) // 3 for msg in self.conversation_history)

        # 如果估算的token数量超过限制，裁剪历史
        if estimated_tokens > max_tokens:
            # 保留系统消息
            system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]

            # 获取非系统消息
            non_system_messages = [msg for msg in self.conversation_history if msg["role"] != "system"]

            # 计算需要保留的非系统消息数量
            # 假设每条消息平均长度相同，计算需要移除的消息数量
            avg_msg_tokens = estimated_tokens / len(self.conversation_history)
            msgs_to_remove = int((estimated_tokens - max_tokens) / avg_msg_tokens) + 1

            # 保留最近的消息，移除较早的消息
            kept_messages = non_system_messages[msgs_to_remove:]

            # 重建对话历史
            self.conversation_history = system_messages + kept_messages

            print(f"对话历史已裁剪，移除了{msgs_to_remove}条较早的消息")

    def ensure_alternating_roles(self, messages: list) -> list:
        """
        确保消息序列中不会出现连续的用户或助手消息

        Args:
            messages: 消息序列

        Returns:
            处理后的消息序列
        """
        if len(messages) <= 1:
            return messages

        result = [messages[0]]  # 保留第一条消息（通常是系统消息）

        for i in range(1, len(messages)):
            current_msg = messages[i]
            prev_msg = result[-1]

            # 如果当前消息和前一条消息角色相同（且不是系统消息）
            if current_msg["role"] == prev_msg["role"] and current_msg["role"] != "system":
                # 插入一个中间消息
                if current_msg["role"] == "user":
                    result.append({"role": "assistant", "content": "我理解您的问题，请继续。"})
                else:  # assistant
                    result.append({"role": "user", "content": "请继续说明。"})

            result.append(current_msg)

        return result

    def get_response(self, message: str, temperature: float = 0.7) -> str:
        """
        获取模型回复

        Args:
            message: 输入的消息
            temperature: 温度参数，控制响应的随机性

        Returns:
            模型的回复
        """
        # 管理对话历史长度
        self.manage_history_length()

        # 检查最后一条消息是否是用户消息，避免连续的用户消息
        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            # 插入一个助手消息，避免连续的用户消息
            self.conversation_history.append({"role": "assistant", "content": "我理解您的问题，请继续。"})

        # 添加用户消息到历史
        self.conversation_history.append({"role": "user", "content": message})

        # 确保系统消息在消息序列的开头
        messages_to_send = []
        system_message_found = False

        # 检查是否已有系统消息，并确保它在第一位
        for msg in self.conversation_history:
            if msg["role"] == "system":
                if not system_message_found:
                    messages_to_send.insert(0, msg)  # 将系统消息放在最前面
                    system_message_found = True
                # 忽略其他系统消息
            else:
                messages_to_send.append(msg)

        # 如果没有系统消息但有系统提示词，添加一个
        if not system_message_found and self.system_prompt:
            messages_to_send.insert(0, {"role": "system", "content": self.system_prompt})

        # 确保消息序列中不会出现连续的用户或助手消息
        messages_to_send = self.ensure_alternating_roles(messages_to_send)

        # 调用API获取回复
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send,
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
        # 管理对话历史长度
        self.manage_history_length()

        # 检查是否会导致连续相同角色的消息
        if self.conversation_history and self.conversation_history[-1]["role"] == role and role != "system":
            # 插入一个中间消息
            if role == "user":
                self.conversation_history.append({"role": "assistant", "content": "我理解您的问题，请继续。"})
            elif role == "assistant":
                self.conversation_history.append({"role": "user", "content": "请继续说明。"})

        # 添加消息到历史
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

        # 构建记忆内容，但限制长度以避免超出上下文限制
        memory_content = "以下是之前阶段的关键总结，请参考这些信息：\n\n"

        # 估计每个记忆的token数量，并限制总量
        max_memory_tokens = 20000  # 为记忆分配的最大token数
        current_tokens = 0

        for memory in memories_to_inject:
            # 估算当前记忆的token数量
            memory_text = f"--- {memory['phase']} 阶段总结 ---\n{memory['content']}\n\n"
            estimated_tokens = len(memory_text) // 3  # 简单估算

            # 如果添加这个记忆会超出限制，则跳过
            if current_tokens + estimated_tokens > max_memory_tokens:
                memory_content += "...(部分较早的记忆已省略)...\n\n"
                break

            memory_content += memory_text
            current_tokens += estimated_tokens

        # 将记忆作为系统消息注入到对话历史中
        # 先清除之前的记忆系统消息，避免重复
        self.conversation_history = [msg for msg in self.conversation_history
                                    if not (msg["role"] == "system" and "阶段总结" in msg.get("content", ""))]

        # 添加新的记忆系统消息
        self.add_message_to_history("system", memory_content)