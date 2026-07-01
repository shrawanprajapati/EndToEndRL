import re

main_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\main.dart"
risk_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens\risk_tab.dart"
ohlcv_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens\ohlcv_tab.dart"
portfolio_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens\portfolio_tab.dart"
live_stream_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens\live_stream_tab.dart"
terminal_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens\terminal_tab.dart"

# 1. Fix main.dart highAlertFired/lowAlertFired to use the declared variables
with open(main_path, "r", encoding="utf-8") as f:
    main_code = f.read()

# Replace any highAlertFired/lowAlertFired with _highAlertFired/_lowAlertFired
main_code = re.sub(r'\bhighAlertFired\b', '_highAlertFired', main_code)
main_code = re.sub(r'\blowAlertFired\b', '_lowAlertFired', main_code)

# 2. Let's make sure buildTimeRangeSelector and buildVolumeProfileBars are defined in main.dart
# Wait, let's extract them from portfolio_tab.dart and put them back in main.dart
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

time_range_method = extract_method("buildTimeRangeSelector", port_code)
vol_profile_method = extract_method("buildVolumeProfileBars", port_code)

# Remove them from portfolio_tab.dart so they are not duplicate
if time_range_method:
    port_code = port_code.replace(time_range_method, "")
if vol_profile_method:
    port_code = port_code.replace(vol_profile_method, "")

with open(portfolio_path, "w", encoding="utf-8") as f:
    f.write(port_code)

# Add them to main.dart as methods of _DashboardHomeScreenState
# We will insert them right before buildDialGauge
if time_range_method:
    # Remove state. prefixes since it's now in DashboardHomeScreenState
    time_range_method = re.sub(r'\bwidget\.state\.', '', time_range_method)
    main_code = main_code.replace("Widget buildDialGauge", time_range_method + "\n\n  Widget buildDialGauge")
if vol_profile_method:
    vol_profile_method = re.sub(r'\bwidget\.state\.', '', vol_profile_method)
    main_code = main_code.replace("Widget buildDialGauge", vol_profile_method + "\n\n  Widget buildDialGauge")

with open(main_path, "w", encoding="utf-8") as f:
    f.write(main_code)

# 3. Fix risk_tab.dart
with open(risk_path, "r", encoding="utf-8") as f:
    risk_code = f.read()
# Replace _applyRiskLimits with widget.state.applyRiskLimits
risk_code = re.sub(r'\b_applyRiskLimits\b', 'widget.state.applyRiskLimits', risk_code)
risk_code = re.sub(r'\bapplyRiskLimits\b', 'widget.state.applyRiskLimits', risk_code)
risk_code = re.sub(r'\bwidget\.state\.widget\.state\.applyRiskLimits\b', 'widget.state.applyRiskLimits', risk_code)
with open(risk_path, "w", encoding="utf-8") as f:
    f.write(risk_code)

# 4. Fix ohlcv_tab.dart
with open(ohlcv_path, "r", encoding="utf-8") as f:
    ohlcv_code = f.read()
# Make sure it calls widget.state.buildTimeRangeSelector() and widget.state.buildVolumeProfileBars()
ohlcv_code = re.sub(r'\bwidget\.state\.buildTimeRangeSelector\b', 'widget.state.buildTimeRangeSelector', ohlcv_code)
ohlcv_code = re.sub(r'\bwidget\.state\.buildVolumeProfileBars\b', 'widget.state.buildVolumeProfileBars', ohlcv_code)
with open(ohlcv_path, "w", encoding="utf-8") as f:
    f.write(ohlcv_code)

# 5. Fix live_stream_tab.dart
with open(live_stream_path, "r", encoding="utf-8") as f:
    live_code = f.read()
# Replace widget.state.widget.state.buildAllocationBarCard with widget.state.buildAllocationBarCard
live_code = live_code.replace("widget.state.widget.state.buildAllocationBarCard", "widget.state.buildAllocationBarCard")
live_code = live_code.replace("widget.state.widget.state.buildActionsDistributionChart", "widget.state.buildActionsDistributionChart")
with open(live_stream_path, "w", encoding="utf-8") as f:
    f.write(live_code)

# 6. Fix terminal_tab.dart
with open(terminal_path, "r", encoding="utf-8") as f:
    term_code = f.read()
term_code = term_code.replace("widget.state.widget.state.reloadModelCheckpoint", "widget.state.reloadModelCheckpoint")
with open(terminal_path, "w", encoding="utf-8") as f:
    f.write(term_code)

print("Applied final compile fixes.")
