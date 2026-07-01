import os
import re

screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"

for filename in os.listdir(screens_dir):
    if not filename.endswith(".dart") or filename == "dashboard.dart":
        continue
    fpath = os.path.join(screens_dir, filename)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace DashboardHomeScreenState with _DashboardHomeScreenState
    content = content.replace("DashboardHomeScreenState", "_DashboardHomeScreenState")
    
    # Replace widget.widget.state.primaryAccent with widget.state.widget.primaryAccent
    content = content.replace("widget.widget.state.primaryAccent", "widget.state.widget.primaryAccent")
    
    # Specific fixes
    if filename == "terminal_tab.dart":
        # Fix fetchData() -> widget.state.fetchData()
        content = re.sub(r'\bfetchData\s*\(', 'widget.state.fetchData(', content)
        # Fix _channel -> widget.state._channel
        content = re.sub(r'\b_channel\b', 'widget.state._channel', content)
        # Fix _reloadModelCheckpoint -> widget.state._reloadModelCheckpoint (if it has underscore)
        content = re.sub(r'\b_reloadModelCheckpoint\b', 'widget.state._reloadModelCheckpoint', content)
        content = re.sub(r'\breloadModelCheckpoint\b', 'widget.state.reloadModelCheckpoint', content)
        
    elif filename == "live_stream_tab.dart":
        # Fix _buildAllocationBarCard -> widget.state._buildAllocationBarCard
        content = re.sub(r'\b_buildAllocationBarCard\b', 'widget.state._buildAllocationBarCard', content)
        # Fix _buildActionsDistributionChart -> widget.state._buildActionsDistributionChart
        content = re.sub(r'\b_buildActionsDistributionChart\b', 'widget.state._buildActionsDistributionChart', content)
        
    elif filename == "portfolio_tab.dart":
        # Fix selectedTimeRange -> widget.state.selectedTimeRange
        content = re.sub(r'\bselectedTimeRange\s*=', 'widget.state.selectedTimeRange =', content)
        
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed compile issues in {filename}")
