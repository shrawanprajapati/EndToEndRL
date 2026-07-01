import re
import os

main_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\main.dart"
screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"
os.makedirs(screens_dir, exist_ok=True)

with open(main_path, "r", encoding="utf-8") as f:
    code = f.read()

# Helper to find method start and end braces
def extract_method(name, content):
    pattern = r'(Widget|TableRow|List<Widget>|void|dynamic|double|String|int|bool)\s+' + re.escape(name) + r'\s*\([^{]*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Method {name} not found")
    
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
        
    return content[start_pos:idx], start_pos, idx

# Let's extract the methods and build the screen files
# 1. Risk tab & helpers
risk_helpers = [
    "buildRiskCommander",
    "buildRiskLimitsControlCard",
    "buildCriticalVolWarningCard",
    "buildSharpeGaugeCard",
    "buildDrawdownHeatMap",
    "buildPositionSizeHeatmap",
    "buildCorrelationMatrixCard",
    "buildRiskIndicatorsBlock"
]

risk_methods_code = []
for h in risk_helpers:
    m_code, start, end = extract_method(h, code)
    # Prefix instance variables/methods with state.
    m_code = re.sub(r'\b(liveRegime|maxDrawdownLimit|maxPositionSizeLimit|stopLossPctLimit|takeProfitPctLimit|summary|agentDrawdown|agentPositions|prices|agentEquity|primaryAccent|applyRiskLimits|buildGlassCard|buildDialGauge|buildMetricRow|buildTooltipMetricRow|buildExpandableGlassCard|errorMessage|isLoading|isFullscreenChart|fullscreenChartWidget|openFullscreen)\b', r'widget.state.\1', m_code)
    m_code = re.sub(r'\bwidget\.state\.widget\.primaryAccent\b', r'widget.state.widget.primaryAccent', m_code)
    risk_methods_code.append(m_code)

risk_tab_content = """import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:fl_chart/fl_chart.dart' as fl;
import '../main.dart';
import '../services/websocket_service.dart';

class RiskTab extends ConsumerStatefulWidget {
  final DashboardHomeScreenState state;
  const RiskTab({super.key, required this.state});

  @override
  ConsumerState<RiskTab> createState() => _RiskTabState();
}

class _RiskTabState extends ConsumerState<RiskTab> {
  @override
  Widget build(BuildContext context) {
    return widget.state.buildRiskCommander();
  }
}

extension RiskTabExtensions on _RiskTabState {
""" + "\n\n".join(risk_methods_code) + "\n}"

with open(os.path.join(screens_dir, "risk_tab.dart"), "w", encoding="utf-8") as f:
    f.write(risk_tab_content)

print("Generated risk_tab.dart")
