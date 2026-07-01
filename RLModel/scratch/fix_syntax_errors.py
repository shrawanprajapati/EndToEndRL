import os
import re

screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"

for filename in os.listdir(screens_dir):
    if not filename.endswith(".dart"):
        continue
    
    fpath = os.path.join(screens_dir, filename)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace declarations like: Widget widget.state.method(...)
    # with: Widget method(...)
    types = ["Widget", "TableRow", "List<Widget>", "void", "dynamic", "double", "String", "int", "bool"]
    for t in types:
        pattern = re.escape(t) + r'\s+widget\.state\.(\w+)\s*\('
        content = re.sub(pattern, t + r' \1(', content)
        
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed declarations in {filename}")
