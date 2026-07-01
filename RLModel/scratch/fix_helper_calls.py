import os
import re

screens_dir = r"c:\Users\shraw\OneDrive\Desktop\EndToEndRL-AdwitCode -v3\EndToEndRL-ShrawanCode - Copy\rl_trading_ui\lib\screens"

refactorings = [
    (r'\b_buildGlassCard\b', 'widget.state.buildGlassCard'),
    (r'\b_buildDialGauge\b', 'widget.state.buildDialGauge'),
    (r'\b_buildMetricRow\b', 'widget.state.buildMetricRow'),
    (r'\b_buildTooltipMetricRow\b', 'widget.state.buildTooltipMetricRow'),
    (r'\b_buildExpandableGlassCard\b', 'widget.state.buildExpandableGlassCard'),
]

for filename in os.listdir(screens_dir):
    if not filename.endswith(".dart"):
        continue
    fpath = os.path.join(screens_dir, filename)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        
    for pattern, replacement in refactorings:
        content = re.sub(pattern, replacement, content)
        
    # Cleanup double prefixes if any got generated
    content = re.sub(r'\bwidget\.state\.widget\.state\.', 'widget.state.', content)
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Refactored helper calls in {filename}")
