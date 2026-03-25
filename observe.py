import os
import argparse
from pathlib import Path

def generate_smart_tree(path, limit=3, indent=""):
    """
    递归生成树状结构字符串
    :param path: 当前路径
    :param limit: 每个目录下显示文件的最大数量
    :param indent: 缩进字符串
    """
    tree_str = ""
    try:
        # 获取当前目录下所有条目并排序
        items = sorted(os.listdir(path))
    except PermissionError:
        return indent + "└── [Permission Denied]\n"

    # 分离文件夹和文件
    dirs = [i for i in items if os.path.isdir(os.path.join(path, i))]
    files = [i for i in items if os.path.isfile(os.path.join(path, i))]

    # 处理文件夹
    for i, d in enumerate(dirs):
        is_last = (i == len(dirs) - 1) and (len(files) == 0)
        connector = "└── " if is_last else "├── "
        tree_str += f"{indent}{connector}📁 {d}/\n"
        # 递归进入子文件夹，增加缩进
        new_indent = indent + ("    " if is_last else "│   ")
        tree_str += generate_smart_tree(os.path.join(path, d), limit, new_indent)

    # 处理文件
    for i, f in enumerate(files):
        if i < limit:
            is_last = (i == len(files) - 1)
            connector = "└── " if is_last else "├── "
            tree_str += f"{indent}{connector}📄 {f}\n"
        elif i == limit:
            tree_str += f"{indent}└── ... (and {len(files) - limit} more files)\n"
            break
            
    return tree_str

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="生成带有文件数量限制的目录树结构")
    parser.add_argument("root", help="要扫描的根目录路径")
    parser.add_argument("-o", "--output", default="directory_tree.txt", help="输出的txt文件名 (默认: directory_tree.txt)")
    parser.add_argument("-l", "--limit", type=int, default=3, help="每个目录下显示文件的最大数量 (默认: 3)")

    args = parser.parse_args()

    # 2. 检查目录是否存在
    root_path = os.path.abspath(args.root)
    if not os.path.isdir(root_path):
        print(f"❌ 错误: 路径 '{args.root}' 不是一个有效的目录。")
        return

    print(f"🔍 正在扫描目录: {root_path}")
    
    # 3. 生成树状结构
    header = f"Project Structure for: {root_path}\n" + "="*50 + "\n"
    content = generate_smart_tree(root_path, limit=args.limit)
    
    # 4. 写入文件
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(header + content)
        print(f"✅ 成功！目录树已输出至: {args.output}")
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")

if __name__ == "__main__":
    main()
