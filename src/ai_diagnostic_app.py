"""
Metastasis-Tracker-AI: AI Diagnostic Support Coordinator
Filename: src/ai_diagnostic_app.py
"""

import os
import glob
from datetime import datetime

class AIDiagnosticSupportApp:
    def __init__(self, docs_dir: str = "docs", reports_dir: str = "reports"):
        """Initializes folder nodes for medical records compilation."""
        self.docs_dir = docs_dir
        self.reports_dir = reports_dir
        self.knowledge_base_summary = ""
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def ingest_documentation_vault(self) -> int:
        """Parses local reference guides to frame diagnostic rules."""
        files = glob.glob(os.path.join(self.docs_dir, "*.md"))
        compiled = []
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                compiled.append(f.read())
        self.knowledge_base_summary = "\n".join(compiled)
        return len(files)

    def process_and_evaluate_metrics(self, metrics: dict) -> dict:
        """Checks real-time voxel parameters against established constraints."""
        peak = metrics.get("peak_density", 0.0)
        total = metrics.get("total_voxels", 0)
        violation = 140.0 <= peak <= 690.0
        urgency = "CRITICAL" if total > 500 else "STABLE"
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_voxels_detected": total,
            "matrix_peak_intensity": peak,
            "chitin_signature_match": "POSITIVE" if violation else "NEGATIVE",
            "clinical_status_urgency": urgency,
            "recommended_action": "Initiate High-Dose Vitamin Loading & Fluid Tracking" if urgency == "CRITICAL" else "Maintain Observation Modality"
        }

    def export_diagnosis_support_file(self, results: dict) -> str:
        """Writes structured diagnostic support files into your local reports repository."""
        slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        rp = os.path.join(self.reports_dir, f"diagnostic_support_log_{slug}.md")
        
        content = f"""# Operation Cancer Moonshot: AI Diagnostic Support Document
Generated on: {results['timestamp']}
Status Classification: **{results['clinical_status_urgency']}**

## 📊 Live Array Analytical Metrics
*   Total Voxel Clusters Active: {results['target_voxels_detected']} voxels
*   Peak Signal Vector Value   : {results['matrix_peak_intensity']:.4f} units
*   Chitin Attenuation Match   : {results['chitin_signature_match']}

## 🧠 Algorithmic Diagnostic Support Guidance
*   Directive Action Blueprint : {results['recommended_action']}
"""
        with open(rp, 'w', encoding='utf-8') as f:
            f.write(content)
        return rp
