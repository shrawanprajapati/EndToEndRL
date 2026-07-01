import os
import re

main_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\main.dart"
screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"

# 1. Update each tab file in screens to be a part of main.dart
for filename in os.listdir(screens_dir):
    if not filename.endswith(".dart") or filename == "dashboard.dart":
        continue
    fpath = os.path.join(screens_dir, filename)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Remove all imports
    content = re.sub(r'import\s+[^;]+;', '', content)
    # Strip leading whitespace/newlines
    content = content.strip()
    # Add part of
    content = "part of '../main.dart';\n\n" + content
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Converted {filename} to part file.")

# 2. Add part declarations to main.dart, and remove the imports of screens
with open(main_path, "r", encoding="utf-8") as f:
    main_code = f.read()

# Remove the screens imports we added
main_code = re.sub(r"import\s+'screens/[^']+\.dart';\n", "", main_code)

# Add part declarations right before the first class/method definition
part_declarations = """
part 'screens/risk_tab.dart';
part 'screens/strategy_tab.dart';
part 'screens/ohlcv_tab.dart';
part 'screens/portfolio_tab.dart';
part 'screens/terminal_tab.dart';
part 'screens/live_stream_tab.dart';
"""

# Let's insert the parts after imports in main.dart
# Find the last import
import_matches = list(re.finditer(r'import\s+[^;]+;', main_code))
if import_matches:
    last_import_end = import_matches[-1].end()
    main_code = main_code[:last_import_end] + part_declarations + main_code[last_import_end:]

with open(main_path, "w", encoding="utf-8") as f:
    f.write(main_code)

print("Updated main.dart with part declarations.")
