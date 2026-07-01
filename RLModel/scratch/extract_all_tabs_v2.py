import re
import os

main_path = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\main.dart"
screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"
os.makedirs(screens_dir, exist_ok=True)

with open(main_path, "r", encoding="utf-8") as f:
    code = f.read()

def extract_method(name, content):
    pattern = r'(Widget|TableRow|List<Widget>|void|dynamic|double|String|int|bool)\s+' + re.escape(name) + r'\s*\([^{]*\{'
    match = re.search(pattern, content)
    if not match:
        # Try with underscore
        pattern = r'(Widget|TableRow|List<Widget>|void|dynamic|double|String|int|bool)\s+' + re.escape("_" + name) + r'\s*\([^{]*\{'
        match = re.search(pattern, content)
        if not match:
            raise ValueError(f"Method {name} or _{name} not found")
    
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

# Define tabs and their helper methods with correct names matching main.dart
tabs_config = {
    "risk_tab.dart": {
        "class": "RiskTab",
        "methods": [
            "buildRiskCommander", "buildRiskLimitsControlCard", "buildCriticalVolWarningCard", 
            "buildSharpeGaugeCard", "buildDrawdownHeatMap", "buildPositionSizeHeatmap", 
            "buildCorrelationMatrixCard", "buildRiskIndicatorsBlock"
        ],
        "stateful": True,
        "live_stream": False
    },
    "strategy_tab.dart": {
        "class": "StrategyTab",
        "methods": [
            "buildStrategyLab", "buildStrategyLiveStatsCard", "buildGradeLegendPanel", 
            "buildGradeTag", "buildMultiEpisodeStatsTable", 
            "buildCumulativeAlphaChart", "buildTradeScatterPlot", "buildParamsSimulationSliders", 
            "buildWinLossSummaryCard", "buildBenchmarkSelectors"
        ],
        "stateful": True,
        "live_stream": True
    },
    "ohlcv_tab.dart": {
        "class": "OHLCVTab",
        "methods": [
            "buildOHLCVChart", "buildIndicatorToggles"
        ],
        "stateful": True,
        "live_stream": False
    },
    "portfolio_tab.dart": {
        "class": "PortfolioTab",
        "methods": [
            "buildPortfolioMonitor", "buildTradePerformanceTable", "buildDrawdownStatsCard", 
            "buildVolumeProfileBars", "buildActionsDistributionChart", "buildHoldingsPieChart", 
            "buildEquityWaterfallCard", "buildAllocationBarCard", "buildTimeRangeSelector"
        ],
        "stateful": True,
        "live_stream": False
    },
    "terminal_tab.dart": {
        "class": "TerminalTab",
        "methods": [
            "buildAdvancedTerminal", "buildApiConnectionPanel", "buildDataManagementButtons", 
            "buildSystemHealthMonitor", "buildModelConfigInspector", "buildModelCheckpointReloadPanel", 
            "buildRawCsvPreviewPanel"
        ],
        "stateful": True,
        "live_stream": False
    },
    "live_stream_tab.dart": {
        "class": "LiveStreamTab",
        "methods": [
            "buildLiveStreamTab", "buildStreamEquityChartWidget", "buildStreamPnlChartWidget", 
            "buildStreamLatencyBadge", "buildStreamLogConsole"
        ],
        "stateful": True,
        "live_stream": True
    }
}

removed_ranges = []

for filename, cfg in tabs_config.items():
    widget_class = cfg["class"]
    methods_code = []
    
    for m in cfg["methods"]:
        try:
            m_code, start, end = extract_method(m, code)
            removed_ranges.append((start, end))
            
            # Clean leading underscore if present in method definition
            m_code = re.sub(r'Widget\s+_' + re.escape(m), r'Widget ' + m, m_code)
            m_code = re.sub(r'TableRow\s+_' + re.escape(m), r'TableRow ' + m, m_code)
            m_code = re.sub(r'List<Widget>\s+_' + re.escape(m), r'List<Widget> ' + m, m_code)
            m_code = re.sub(r'void\s+_' + re.escape(m), r'void ' + m, m_code)
            
            # Prefix variables from DashboardHomeScreenState with widget.state.
            state_fields = [
                "liveRegime", "maxDrawdownLimit", "maxPositionSizeLimit", "stopLossPctLimit", "takeProfitPctLimit",
                "summary", "agentDrawdown", "agentPositions", "prices", "agentEquity", "primaryAccent", "applyRiskLimits",
                "buildGlassCard", "buildDialGauge", "buildMetricRow", "buildTooltipMetricRow", "buildExpandableGlassCard",
                "errorMessage", "isLoading", "isFullscreenChart", "fullscreenChartWidget", "openFullscreen", "backtestReport",
                "timestamps", "tradeLog", "tradeSearchQuery", "tradeTypeFilter", "slippageParam", "commissionParam",
                "showEMALine", "showBollingerBands", "priceAlertHigh", "priceAlertLow", "priceAlertHighController",
                "priceAlertLowController", "streamEquitySpots", "streamBhSpots", "streamPnlSpots", "streamTerminalLogs",
                "streamScrollController", "initialStreamBhPrice", "streamTickCount", "lastStreamTimestamp", "streamLatencyMs",
                "lastRegimeState", "regimeTransitions", "lastProcessedStreamTimestamp", "lastProcessedStreamLog",
                "liveDataQuality", "isEmergencyStopped", "activeCheckpoint", "isFullscreen", "activeTab", "apiUrl",
                "jsonTextController", "apiTextController", "updateAccent", "toggleTheme", "loadFromData", "buildBenchmarkSelectors",
                "gradeLegendPanel", "buildGradeLegendPanel", "buildGradeTag", "buildStrategyLiveStatsCard"
            ]
            
            # Make sure we replace calls to other extracted methods too
            for other_m in cfg["methods"]:
                m_code = re.sub(r'\b_' + re.escape(other_m) + r'\b', other_m, m_code)
                m_code = re.sub(r'\b' + re.escape(other_m) + r'\b', other_m, m_code)
                
            for field in state_fields:
                m_code = re.sub(r'\b' + re.escape(field) + r'\b', r'widget.state.' + field, m_code)
                m_code = re.sub(r'\bwidget\.state\.widget\.state\.', r'widget.state.', m_code)
                m_code = re.sub(r'\bwidget\.state\.widget\.', r'widget.state.widget.', m_code)
                
            methods_code.append(m_code)
        except Exception as ex:
            print(f"Error extracting {m} for {filename}: {ex}")

    live_stream_param = "final LiveStreamData? liveStream;" if cfg["live_stream"] else ""
    live_stream_init = "required this.liveStream," if cfg["live_stream"] else ""
    live_stream_widget_param = "widget.liveStream" if cfg["live_stream"] else ""
    
    file_content = f"""import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:fl_chart/fl_chart.dart' as fl;
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:candlesticks/candlesticks.dart';
import '../main.dart';
import '../services/websocket_service.dart';
import '../providers.dart';

class {widget_class} extends ConsumerStatefulWidget {{
  final DashboardHomeScreenState state;
  {live_stream_param}
  const {widget_class}({{super.key, required this.state, {live_stream_init}}});

  @override
  ConsumerState<{widget_class}> createState() => _{widget_class}State();
}}

class _{widget_class}State extends ConsumerState<{widget_class}> {{
  @override
  Widget build(BuildContext context) {{
    return widget.state.{cfg["methods"][0]}({live_stream_widget_param});
  }}
}}

extension {widget_class}Extensions on _{widget_class}State {{
""" + "\n\n".join(methods_code) + "\n}"

    with open(os.path.join(screens_dir, filename), "w", encoding="utf-8") as f:
        f.write(file_content)
    print(f"Generated {filename}")

# Now remove the extracted methods from main.dart
removed_ranges.sort(key=lambda x: x[0], reverse=True)
for start, end in removed_ranges:
    code = code[:start] + code[end:]

# Replace the calls in buildActiveTabContent inside main.dart
code = re.sub(r'case 0:\s*return\s+_buildRiskCommander\(\);', r'case 0: return RiskTab(state: this);', code)
code = re.sub(r'case 1:\s*return\s+_buildStrategyLab\(liveStream\);', r'case 1: return StrategyTab(state: this, liveStream: liveStream);', code)
code = re.sub(r'case 2:\s*return\s+_buildOHLCVChart\(\);', r'case 2: return OHLCVTab(state: this);', code)
code = re.sub(r'case 3:\s*return\s+_buildPortfolioMonitor\(\);', r'case 3: return PortfolioTab(state: this);', code)
code = re.sub(r'case 4:\s*return\s+_buildAdvancedTerminal\(\);', r'case 4: return TerminalTab(state: this);', code)
code = re.sub(r'case 5:\s*return\s+_buildLiveStreamTab\(liveStream\);', r'case 5: return LiveStreamTab(state: this, liveStream: liveStream);', code)

imports = """import 'screens/risk_tab.dart';
import 'screens/strategy_tab.dart';
import 'screens/ohlcv_tab.dart';
import 'screens/portfolio_tab.dart';
import 'screens/terminal_tab.dart';
import 'screens/live_stream_tab.dart';
import 'providers.dart';
"""

code = imports + code

with open(main_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated main.dart and removed extracted methods.")
