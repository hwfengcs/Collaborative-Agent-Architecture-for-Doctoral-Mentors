import json
import os
import argparse
from typing import List, Dict, Any

def load_paper_drafts(filename: str = "paper_drafts.json") -> List[Dict[str, Any]]:
    """
    加载论文草稿历史
    
    Args:
        filename: 论文草稿JSON文件名
    
    Returns:
        论文草稿历史列表
    """
    if not os.path.exists(filename):
        print(f"错误：找不到文件 {filename}")
        return []
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件出错: {str(e)}")
        return []

def load_final_paper(filename: str = "final_paper.md") -> str:
    """
    加载最终论文
    
    Args:
        filename: 最终论文文件名
    
    Returns:
        论文内容
    """
    if not os.path.exists(filename):
        print(f"错误：找不到文件 {filename}")
        return ""
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"加载文件出错: {str(e)}")
        return ""

def show_paper_versions(drafts: List[Dict[str, Any]]) -> None:
    """
    显示所有论文版本
    
    Args:
        drafts: 论文草稿历史列表
    """
    if not drafts:
        print("没有找到论文草稿记录")
        return
    
    print("\n论文版本列表：")
    print("-" * 50)
    for draft in drafts:
        print(f"版本: {draft['version']}")
        print(f"阶段: {draft['phase']}")
        print(f"内容长度: {len(draft['content'])} 字符")
        print("-" * 50)

def view_paper_version(drafts: List[Dict[str, Any]], version: int) -> None:
    """
    查看指定版本的论文
    
    Args:
        drafts: 论文草稿历史列表
        version: 要查看的版本号
    """
    if not drafts:
        print("没有找到论文草稿记录")
        return
    
    for draft in drafts:
        if draft['version'] == version:
            print(f"\n查看论文 - 版本 {version} (阶段: {draft['phase']})")
            print("=" * 80)
            print(draft['content'])
            print("=" * 80)
            return
    
    print(f"未找到版本 {version} 的论文")

def main():
    parser = argparse.ArgumentParser(description="查看博士生论文")
    parser.add_argument("--list", action="store_true", help="列出所有论文版本")
    parser.add_argument("--version", type=int, help="查看指定版本的论文")
    parser.add_argument("--final", action="store_true", help="查看最终论文")
    parser.add_argument("--drafts-file", default="paper_drafts.json", help="论文草稿文件路径")
    parser.add_argument("--final-file", default="final_paper.md", help="最终论文文件路径")
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，默认查看最终论文
    if not (args.list or args.version is not None or args.final):
        args.final = True
    
    # 加载论文草稿历史
    drafts = []
    if args.list or args.version is not None:
        drafts = load_paper_drafts(args.drafts_file)
    
    # 列出所有版本
    if args.list:
        show_paper_versions(drafts)
    
    # 查看指定版本
    if args.version is not None:
        view_paper_version(drafts, args.version)
    
    # 查看最终论文
    if args.final:
        final_paper = load_final_paper(args.final_file)
        if final_paper:
            print("\n最终论文")
            print("=" * 80)
            print(final_paper)
            print("=" * 80)

if __name__ == "__main__":
    main() 