"""Report generation for model analysis results."""

import base64
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .plots import create_summary_plots


class ReportGenerator:
    """Generate analysis reports in various formats."""

    def __init__(
        self,
        template_dir: Optional[Union[str, Path]] = None,
        style: str = "default",
    ) -> None:
        """
        Initialize ReportGenerator.

        Parameters
        ----------
        template_dir : Optional[Union[str, Path]], default=None
            Directory containing templates. If None, uses built-in templates
        style : str, default="default"
            Report style to use
        """
        self.style = style
        
        # Set up template directory
        if template_dir is None:
            # Use built-in templates
            self.template_dir = Path(__file__).parent / "templates"
        else:
            self.template_dir = Path(template_dir)
        
        # Initialize Jinja2 environment
        self._init_template_env()

    def _init_template_env(self) -> None:
        """Initialize Jinja2 template environment."""
        # For now, we'll use string templates
        # In production, these would be separate template files
        self.env = Environment(
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_html_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Union[str, Path],
        include_plots: bool = True,
        plot_format: str = "png",
    ) -> str:
        """
        Generate HTML report from analysis results.

        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Analysis results from analyzer
        output_path : Union[str, Path]
            Output file path
        include_plots : bool, default=True
            Whether to include plots in the report
        plot_format : str, default="png"
            Format for embedded plots

        Returns
        -------
        str
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate plots if requested
        plot_data = {}
        if include_plots:
            plots = create_summary_plots(analysis_results)
            plot_data = self._encode_plots(plots, plot_format)

        # Prepare template data
        template_data = self._prepare_template_data(analysis_results, plot_data)

        # Render template
        html_content = self._render_html_template(template_data)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(output_path)

    def _encode_plots(
        self,
        plots: Dict[str, Any],
        format: str = "png"
    ) -> Dict[str, str]:
        """
        Encode plots as base64 strings for embedding.

        Parameters
        ----------
        plots : Dict[str, Any]
            Dictionary of plot figures
        format : str, default="png"
            Image format

        Returns
        -------
        Dict[str, str]
            Base64 encoded plot data
        """
        encoded_plots = {}
        
        for name, fig in plots.items():
            # Save figure to bytes buffer
            buffer = io.BytesIO()
            fig.savefig(buffer, format=format, bbox_inches="tight", dpi=100)
            buffer.seek(0)
            
            # Encode as base64
            img_str = base64.b64encode(buffer.read()).decode()
            encoded_plots[name] = f"data:image/{format};base64,{img_str}"
            
            # Close figure to free memory
            plt.close(fig)
        
        return encoded_plots

    def _prepare_template_data(
        self,
        analysis_results: Dict[str, Any],
        plot_data: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Prepare data for template rendering.

        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Analysis results
        plot_data : Dict[str, str]
            Encoded plot data

        Returns
        -------
        Dict[str, Any]
            Template context data
        """
        # Extract key information
        model_info = analysis_results.get("model_info", {})
        feature_importance = analysis_results.get("feature_importance", {})
        performance = analysis_results.get("performance_metrics", {})
        overfitting = analysis_results.get("overfitting_analysis", {})
        suggestions = analysis_results.get("suggestions", [])

        # Prepare template context
        context = {
            "title": "Model Analysis Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_info.get("model_type", "Unknown"),
            "model_info": model_info,
            "plots": plot_data,
            "performance": performance,
            "feature_analysis": {
                "top_features": feature_importance.get("top_features", []),
                "low_importance_count": len(
                    feature_importance.get("low_importance_features", [])
                ),
                "total_features": feature_importance.get(
                    "cumulative_importance", {}
                ).get("total_features", 0),
                "features_for_90pct": feature_importance.get(
                    "cumulative_importance", {}
                ).get("features_for_90pct", 0),
            },
            "overfitting": overfitting,
            "suggestions": suggestions,
            "has_validation": overfitting.get("has_validation", False),
        }

        return context

    def _render_html_template(self, context: Dict[str, Any]) -> str:
        """
        Render HTML template with context data.

        Parameters
        ----------
        context : Dict[str, Any]
            Template context

        Returns
        -------
        str
            Rendered HTML content
        """
        # For simplicity, using inline template
        # In production, this would be loaded from a file
        template_str = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            margin-top: 30px;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .info-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .suggestion {
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .suggestion-high {
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p><strong>Generated:</strong> {{ generated_at }}</p>
        
        <h2>Model Overview</h2>
        <div class="info-grid">
            <div class="info-card">
                <div class="metric-label">Model Type</div>
                <div class="metric-value">{{ model_type|upper }}</div>
            </div>
            <div class="info-card">
                <div class="metric-label">Number of Features</div>
                <div class="metric-value">{{ model_info.n_features|default(0) }}</div>
            </div>
            <div class="info-card">
                <div class="metric-label">Task Type</div>
                <div class="metric-value">
                    {% if model_info.is_classifier %}Classification{% else %}Regression{% endif %}
                </div>
            </div>
            {% if model_info.n_iterations %}
            <div class="info-card">
                <div class="metric-label">Iterations/Trees</div>
                <div class="metric-value">{{ model_info.n_iterations }}</div>
            </div>
            {% endif %}
        </div>
        
        {% if performance %}
        <h2>Performance Metrics</h2>
        {% if plots.performance_metrics %}
        <div class="plot-container">
            <img src="{{ plots.performance_metrics }}" alt="Performance Metrics">
        </div>
        {% endif %}
        
        {% if performance.train %}
        <h3>Training Performance</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {% for metric, value in performance.train.items() %}
            <tr>
                <td>{{ metric|upper }}</td>
                <td>{{ "%.4f"|format(value) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if performance.validation %}
        <h3>Validation Performance</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {% for metric, value in performance.validation.items() %}
            <tr>
                <td>{{ metric|upper }}</td>
                <td>{{ "%.4f"|format(value) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        {% endif %}
        
        <h2>Feature Analysis</h2>
        <div class="info-grid">
            <div class="info-card">
                <div class="metric-label">Total Features</div>
                <div class="metric-value">{{ feature_analysis.total_features }}</div>
            </div>
            <div class="info-card">
                <div class="metric-label">Features for 90% Importance</div>
                <div class="metric-value">{{ feature_analysis.features_for_90pct }}</div>
            </div>
            <div class="info-card">
                <div class="metric-label">Low Importance Features</div>
                <div class="metric-value">{{ feature_analysis.low_importance_count }}</div>
            </div>
        </div>
        
        {% if plots.feature_importance %}
        <h3>Feature Importance</h3>
        <div class="plot-container">
            <img src="{{ plots.feature_importance }}" alt="Feature Importance">
        </div>
        {% endif %}
        
        {% if plots.cumulative_importance %}
        <h3>Cumulative Feature Importance</h3>
        <div class="plot-container">
            <img src="{{ plots.cumulative_importance }}" alt="Cumulative Importance">
        </div>
        {% endif %}
        
        {% if feature_analysis.top_features %}
        <h3>Top Features</h3>
        <ol>
            {% for feature in feature_analysis.top_features[:10] %}
            <li>{{ feature }}</li>
            {% endfor %}
        </ol>
        {% endif %}
        
        {% if has_validation and overfitting %}
        <h2>Overfitting Analysis</h2>
        {% if plots.overfitting_diagnosis %}
        <div class="plot-container">
            <img src="{{ plots.overfitting_diagnosis }}" alt="Overfitting Diagnosis">
        </div>
        {% endif %}
        
        <div class="info-grid">
            <div class="info-card">
                <div class="metric-label">Overfitting Detected</div>
                <div class="metric-value">
                    {% if overfitting.overfitting_detected %}YES{% else %}NO{% endif %}
                </div>
            </div>
            {% if overfitting.severity %}
            <div class="info-card">
                <div class="metric-label">Severity</div>
                <div class="metric-value">{{ overfitting.severity|upper }}</div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if suggestions %}
        <h2>Improvement Suggestions</h2>
        {% for suggestion in suggestions %}
        <div class="suggestion {% if suggestion.priority == 'high' %}suggestion-high{% endif %}">
            <h4>{{ suggestion.type|title }} (Priority: {{ suggestion.priority|upper }})</h4>
            <p><strong>{{ suggestion.message }}</strong></p>
            {% if suggestion.recommendations %}
            <ul>
                {% for rec in suggestion.recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </div>
        {% endfor %}
        {% endif %}
        
        <div class="footer">
            <p>Generated by Model Improvement Toolkit</p>
        </div>
    </div>
</body>
</html>
'''
        
        template = self.env.from_string(template_str)
        return template.render(**context)

    def generate_summary_text(
        self,
        analysis_results: Dict[str, Any]
    ) -> str:
        """
        Generate text summary of analysis results.

        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Analysis results

        Returns
        -------
        str
            Text summary
        """
        lines = []
        lines.append("=== MODEL ANALYSIS SUMMARY ===\n")
        
        # Model info
        model_info = analysis_results.get("model_info", {})
        lines.append(f"Model Type: {model_info.get('model_type', 'Unknown').upper()}")
        lines.append(f"Features: {model_info.get('n_features', 'Unknown')}")
        lines.append(f"Task: {'Classification' if model_info.get('is_classifier') else 'Regression'}")
        
        # Performance
        performance = analysis_results.get("performance_metrics", {})
        if performance.get("train"):
            lines.append("\nTraining Performance:")
            for metric, value in performance["train"].items():
                lines.append(f"  {metric}: {value:.4f}")
        
        if performance.get("validation"):
            lines.append("\nValidation Performance:")
            for metric, value in performance["validation"].items():
                lines.append(f"  {metric}: {value:.4f}")
        
        # Overfitting
        overfitting = analysis_results.get("overfitting_analysis", {})
        if overfitting.get("overfitting_detected"):
            lines.append(f"\nOverfitting: DETECTED (Severity: {overfitting.get('severity', 'unknown')})")
        
        # Suggestions
        suggestions = analysis_results.get("suggestions", [])
        if suggestions:
            lines.append("\nTop Suggestions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                lines.append(f"{i}. {suggestion['message']} (Priority: {suggestion['priority']})")
        
        return "\n".join(lines)