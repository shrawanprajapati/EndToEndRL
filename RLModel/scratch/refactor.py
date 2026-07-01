import re

main_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\main.dart"

with open(main_path, "r", encoding="utf-8") as f:
    code = f.read()

# Make _DashboardHomeScreenState public
code = re.sub(r'\b_DashboardHomeScreenState\b', 'DashboardHomeScreenState', code)

# Make other private methods/vars public if they are used by screens
private_renames = [
    ("_applyRiskLimits", "applyRiskLimits"),
    ("_buildGlassCard", "buildGlassCard"),
    ("_buildDialGauge", "buildDialGauge"),
    ("_buildMetricRow", "buildMetricRow"),
    ("_buildTooltipMetricRow", "buildTooltipMetricRow"),
    ("_buildExpandableGlassCard", "buildExpandableGlassCard"),
    ("_lastProcessedStreamTimestamp", "lastProcessedStreamTimestamp"),
    ("_lastProcessedStreamLog", "lastProcessedStreamLog"),
    ("_streamScrollController", "streamScrollController"),
    ("_openFullscreen", "openFullscreen"),
]

for old, new in private_renames:
    code = re.sub(r'\b' + old + r'\b', new, code)

with open(main_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Renamed private methods and state class in main.dart")
