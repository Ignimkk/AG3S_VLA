from common.workspace import workspace_path as _ag3s_workspace_path
import json,sys
a=json.load(open(_ag3s_workspace_path('outputs/verify/T43/GD/analysis/analysis_V7.json'))); b=json.load(open(_ag3s_workspace_path('outputs/verify/T43/GD/analysis/analysis_V7g.json')))
a["runs"].update(b["runs"]); a["roots"].update(b["roots"])
json.dump(a,open(_ag3s_workspace_path('outputs/verify/T43/GD/analysis/analysis.json'),'w'),indent=1,default=float)
print({k:sum(1 for r in a["runs"] if r.startswith(k+"/")) for k in ("V7","V7g")})
