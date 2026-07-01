import re

main_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\main.dart"
portfolio_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens\portfolio_tab.dart"
risk_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens\risk_tab.dart"

with open(main_path, "r", encoding="utf-8") as f:
    main_code = f.read()

with open(portfolio_path, "r", encoding="utf-8") as f:
    port_code = f.read()

def extract_method(name, content):
    pattern = r'(Widget|TableRow|List<Widget>|void|dynamic|double|String|int|bool)\s+' + re.escape(name) + r'\s*\([^{]*\{'
    match = re.search(pattern, content)
    if not match:
        return None
    start_pos = match.start()
    brace_count = 1
    idx = content.find('{', start_pos) + 1
    while brace_count > 0 and idx < len(content):
        char = content[idx]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        idx += 1
    return content[start_pos:idx]

alloc_method = extract_method("buildAllocationBarCard", port_code)
actions_method = extract_method("buildActionsDistributionChart", port_code)

# Remove from portfolio_tab.dart
if alloc_method:
    port_code = port_code.replace(alloc_method, "")
if actions_method:
    port_code = port_code.replace(actions_method, "")

with open(portfolio_path, "w", encoding="utf-8") as f:
    f.write(port_code)

# Add to main.dart
if alloc_method:
    alloc_method = re.sub(r'\bwidget\.state\.', '', alloc_method)
    main_code = main_code.replace("Widget buildDialGauge", alloc_method + "\n\n  Widget buildDialGauge")
if actions_method:
    actions_method = re.sub(r'\bwidget\.state\.', '', actions_method)
    main_code = main_code.replace("Widget buildDialGauge", actions_method + "\n\n  Widget buildDialGauge")

with open(main_path, "w", encoding="utf-8") as f:
    f.write(main_code)

# Fix risk_tab.dart applyRiskLimits to _applyRiskLimits
with open(risk_path, "r", encoding="utf-8") as f:
    risk_code = f.read()

risk_code = risk_code.replace("widget.state.applyRiskLimits", "widget.state._applyRiskLimits")
risk_code = risk_code.replace("widget.state._applyRiskLimits", "widget.state._applyRiskLimits")

with open(risk_path, "w", encoding="utf-8") as f:
    f.write(risk_code)

print("Moved shared methods to main.dart and fixed risk_tab.dart")
