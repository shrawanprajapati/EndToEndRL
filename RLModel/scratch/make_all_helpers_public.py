import os
import re

main_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\main.dart"
screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"

# 1. Read main.dart and find all private methods and fields starting with _ (except flutter standard ones like _DashboardHomeScreenState, _TradingDashboardAppState)
with open(main_path, "r", encoding="utf-8") as f:
    main_code = f.read()

# Let's find all private methods/fields that are defined in main.dart
# e.g., _buildHeader, _getFilteredCandles, _reloadModelCheckpoint, etc.
private_members = re.findall(r'\b(_[a-zA-Z0-9]+)\b', main_code)
# Filter out system classes/standard names
filtered_members = []
for m in set(private_members):
    if m in ["_TradingDashboardAppState", "_RiskTabState", "_StrategyTabState", "_OHLCVTabState", "_PortfolioTabState", "_TerminalTabState", "_LiveStreamTabState"]:
        continue
    # Only keep things starting with _build, _get, _reload, _connect, _last, _highAlert, _lowAlert, or fields
    if any(m.startswith(prefix) for prefix in ["_build", "_get", "_reload", "_connect", "_last", "_highAlert", "_lowAlert", "_stream"]):
        filtered_members.append(m)

print(f"Found {len(filtered_members)} private members to make public.")

# 2. Make them public in main.dart
for m in filtered_members:
    public_name = m[1:] # strip leading underscore
    main_code = re.sub(r'\b' + re.escape(m) + r'\b', public_name, main_code)

with open(main_path, "w", encoding="utf-8") as f:
    f.write(main_code)

print("Updated main.dart")

# 3. For all files in screens, update private member calls to use widget.state.public_name
for filename in os.listdir(screens_dir):
    if not filename.endswith(".dart") or filename == "dashboard.dart":
        continue
    fpath = os.path.join(screens_dir, filename)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        
    for m in filtered_members:
        public_name = m[1:]
        # Replace occurrences like _buildHeader(...) with widget.state.buildHeader(...)
        content = re.sub(r'\b' + re.escape(m) + r'\b', r'widget.state.' + public_name, content)
        content = re.sub(r'\b' + re.escape(public_name) + r'\b', r'widget.state.' + public_name, content)
        
    # Clean up double prefixes like widget.state.widget.state.
    content = re.sub(r'\bwidget\.state\.widget\.state\.', 'widget.state.', content)
    content = re.sub(r'\bwidget\.state\.widget\.', 'widget.state.widget.', content)
    # Restore local extension methods (meaning they should not be prefixed with widget.state. in their declaration)
    # We can detect declarations: widget.state.Widget method(...) and change to Widget method(...)
    types = ["Widget", "TableRow", "List<Widget>", "void", "dynamic", "double", "String", "int", "bool"]
    for t in types:
        pattern = r'widget\.state\.' + re.escape(t) + r'\s+(\w+)\s*\('
        content = re.sub(pattern, t + r' \1(', content)
        # Also declarations starting with type followed by widget.state.method
        pattern = re.escape(t) + r'\s+widget\.state\.(\w+)\s*\('
        content = re.sub(pattern, t + r' \1(', content)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filename}")
