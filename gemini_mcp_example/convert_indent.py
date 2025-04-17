# convert_indent.py

import sys


def convert_indent(file_path, old_indent=4, new_indent=2):
  with open(file_path, "r") as f:
    lines = f.readlines()

  converted_lines = []
  for line in lines:
    leading_spaces = len(line) - len(line.lstrip(" "))
    if leading_spaces % old_indent != 0:
      # スキップまたは警告（不正なインデント）
      converted_lines.append(line)
      continue
    indent_level = leading_spaces // old_indent
    new_indent_str = " " * (indent_level * new_indent)
    converted_lines.append(new_indent_str + line.lstrip(" "))

  with open(file_path, "w") as f:
    f.writelines(converted_lines)

  print(f"Converted {file_path} from {old_indent}-space to {new_indent}-space indentation.")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python convert_indent.py <file.py>")
    sys.exit(1)
  convert_indent(sys.argv[1])
