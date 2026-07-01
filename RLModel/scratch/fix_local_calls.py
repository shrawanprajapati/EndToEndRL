import os
import re

screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"

# Define tabs and their local methods
tabs_config = {
    "risk_tab.dart": [
        "buildRiskCommander", "buildRiskLimitsControlCard", "buildCriticalVolWarningCard", 
        "buildSharpeGaugeCard", "buildDrawdownHeatMap", "buildPositionSizeHeatmap", 
        "buildCorrelationMatrixCard", "buildRiskIndicatorsBlock"
    ],
    "strategy_tab.dart": [
        "buildStrategyLab", "buildStrategyLiveStatsCard", "buildGradeLegendPanel", 
        "buildGradeTag", "buildMultiEpisodeStatsTable", 
        "buildCumulativeAlphaChart", "buildTradeScatterPlot", "buildParamsSimulationSliders", 
        "buildWinLossSummaryCard", "buildBenchmarkSelectors"
    ],
    "ohlcv_tab.dart": [
        "buildOHLCVChart", "buildIndicatorToggles"
    ],
    "portfolio_tab.dart": [
        "buildPortfolioMonitor", "buildTradePerformanceTable", "buildDrawdownStatsCard", 
        "buildVolumeProfileBars", "buildActionsDistributionChart", "buildHoldingsPieChart", 
        "buildEquityWaterfallCard", "buildAllocationBarCard", "buildTimeRangeSelector"
    ],
    "terminal_tab.dart": [
        "buildAdvancedTerminal", "buildApiConnectionPanel", "buildDataManagementButtons", 
        "buildSystemHealthMonitor", "buildModelConfigInspector", "buildModelCheckpointReloadPanel", 
        "buildRawCsvPreviewPanel"
    ],
    "live_stream_tab.dart": [
        "buildLiveStreamTab", "buildStreamEquityChartWidget", "buildStreamPnlChartWidget", 
        "buildStreamLatencyBadge", "buildStreamLogConsole"
    ]
}

for filename, local_methods in tabs_config.items():
    fpath = os.path.join(screens_dir, filename)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 1. In build() method, call the main method directly:
    # e.g. return widget.state.buildRiskCommander(); -> return buildRiskCommander();
    main_method = local_methods[0]
    content = re.sub(r'widget\.state\.' + re.escape(main_method) + r'\b', main_method, content)
    
    # 2. For all local methods, remove the widget.state. prefix
    for m in local_methods:
        content = re.sub(r'\bwidget\.state\.' + re.escape(m) + r'\b', m, content)
        
    # Additional specific fixes:
    if filename == "portfolio_tab.dart":
        # Fix _exportTradesToCSV -> widget.state._exportTradesToCSV
        content = re.sub(r'\b_exportTradesToCSV\b', 'widget.state._exportTradesToCSV', content)
        # Fix streamActionHistory -> widget.state.streamActionHistory
        content = re.sub(r'\bstreamActionHistory\b', 'widget.state.streamActionHistory', content)
        
    elif filename == "terminal_tab.dart":
        # Fix widget.state.widget.state.reloadModelCheckpoint -> widget.state.reloadModelCheckpoint
        content = re.sub(r'\bwidget\.widget\.state\.reloadModelCheckpoint\b', 'widget.state.reloadModelCheckpoint', content)
        content = re.sub(r'\bwidget\.state\.widget\.state\.reloadModelCheckpoint\b', 'widget.state.reloadModelCheckpoint', content)
        
    elif filename == "ohlcv_tab.dart":
        # Fix _buildTimeRangeSelector() -> widget.state.buildTimeRangeSelector()
        content = re.sub(r'\b_buildTimeRangeSelector\b', 'widget.state.buildTimeRangeSelector', content)
        # Fix _buildVolumeProfileBars() -> widget.state.buildVolumeProfileBars()
        content = re.sub(r'\b_buildVolumeProfileBars\b', 'widget.state.buildVolumeProfileBars', content)
        # Fix _buildPriceAlertsConfig() -> widget.state.buildPriceAlertsConfig()
        content = re.sub(r'\b_buildPriceAlertsConfig\b', 'widget.state.buildPriceAlertsConfig', content)
        # Fix _getFilteredCandles -> widget.state.getFilteredCandles
        content = re.sub(r'\b_getFilteredCandles\b', 'widget.state.getFilteredCandles', content)
        
    elif filename == "live_stream_tab.dart":
        # Fix _buildAllocationBarCard -> widget.state.buildAllocationBarCard
        content = re.sub(r'\b_buildAllocationBarCard\b', 'widget.state.buildAllocationBarCard', content)
        # Fix _buildActionsDistributionChart -> widget.state.buildActionsDistributionChart
        content = re.sub(r'\b_buildActionsDistributionChart\b', 'widget.state.buildActionsDistributionChart', content)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed local calls in {filename}")
