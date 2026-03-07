#!/usr/bin/env python3
"""
PROJECT RESIDUE - Comparative Performance Analysis
===============================================

Analyze and compare performance results across different test runs
Calculate savings, improvements, and statistical significance
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class PerformanceAnalyzer:
    """Analyze and compare PROJECT RESIDUE performance results"""

    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        print("=== PERFORMANCE ANALYZER ===")
        print("Ready to analyze test results")
        print("=" * 50)

    def load_test_results(self, pattern="*.json"):
        """Load all test result files"""

        result_files = list(self.results_dir.glob(pattern))
        results = []

        print(f"Loading {len(result_files)} result files...")

        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['_filename'] = file_path.name
                    results.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"✅ Loaded {len(results)} test results")
        return results

    def analyze_residue_performance(self, results):
        """Analyze PROJECT RESIDUE specific performance metrics"""

        residue_tests = [r for r in results if r.get('test_type') == 'residue_inference']

        if not residue_tests:
            return {"error": "No RESIDUE performance tests found"}

        analysis = {
            "test_type": "residue_performance_analysis",
            "num_tests": len(residue_tests),
            "metrics": {}
        }

        # Extract timing data
        inference_times = []
        throughputs = []
        scalings = []

        for test in residue_tests:
            for benchmark in test.get('benchmarks', []):
                inference_times.append(benchmark['inference_time'])
                throughputs.append(benchmark['throughput'])
                scalings.append(benchmark['scaling_factor'])

        # Statistical analysis
        analysis["metrics"]["inference_time"] = {
            "mean": np.mean(inference_times),
            "std": np.std(inference_times),
            "min": np.min(inference_times),
            "max": np.max(inference_times),
            "median": np.median(inference_times)
        }

        analysis["metrics"]["throughput"] = {
            "mean": np.mean(throughputs),
            "std": np.std(throughputs),
            "min": np.min(throughputs),
            "max": np.max(throughputs),
            "median": np.median(throughputs)
        }

        analysis["metrics"]["scaling"] = {
            "mean": np.mean(scalings),
            "std": np.std(scalings),
            "min": np.min(scalings),
            "max": np.max(scalings),
            "median": np.median(scalings)
        }

        # Performance classification
        avg_throughput = analysis["metrics"]["throughput"]["mean"]
        if avg_throughput > 10000:
            performance_class = "Excellent"
        elif avg_throughput > 5000:
            performance_class = "Very Good"
        elif avg_throughput > 2000:
            performance_class = "Good"
        else:
            performance_class = "Needs Optimization"

        analysis["performance_classification"] = performance_class

        return analysis

    def analyze_baseline_comparison(self, results):
        """Analyze baseline vs RESIDUE comparison tests"""

        comparison_tests = [r for r in results if r.get('test_type') == 'baseline_comparison']

        if not comparison_tests:
            return {"error": "No baseline comparison tests found"}

        analysis = {
            "test_type": "baseline_comparison_analysis",
            "num_tests": len(comparison_tests),
            "savings_analysis": {}
        }

        # Extract savings data
        savings_percentages = []
        time_improvements = []

        for test in comparison_tests:
            for comp in test.get('comparisons', []):
                savings_percentages.append(comp['savings_percent'])
                time_improvements.append(comp['throughput_improvement'])

        # Statistical analysis of savings
        analysis["savings_analysis"]["savings_percentage"] = {
            "mean": np.mean(savings_percentages),
            "std": np.std(savings_percentages),
            "min": np.min(savings_percentages),
            "max": np.max(savings_percentages),
            "median": np.median(savings_percentages)
        }

        analysis["savings_analysis"]["throughput_improvement"] = {
            "mean": np.mean(time_improvements),
            "std": np.std(time_improvements),
            "min": np.min(time_improvements),
            "max": np.max(time_improvements),
            "median": np.median(time_improvements)
        }

        # Significance testing
        if len(savings_percentages) >= 2:
            # Test if savings are significantly different from zero
            t_stat, p_value = stats.ttest_1samp(savings_percentages, 0)
            analysis["statistical_significance"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_improvement": p_value < 0.05,
                "confidence_level": "95%" if p_value < 0.05 else "Not significant"
            }

        # Business impact assessment
        avg_savings = analysis["savings_analysis"]["savings_percentage"]["mean"]
        if avg_savings > 30:
            business_impact = "Excellent - Major cost reduction"
        elif avg_savings > 20:
            business_impact = "Very Good - Significant savings"
        elif avg_savings > 10:
            business_impact = "Good - Noticeable improvement"
        else:
            business_impact = "Modest - Some benefit"

        analysis["business_impact"] = business_impact

        return analysis

    def analyze_memory_usage(self, results):
        """Analyze memory consumption patterns"""

        memory_tests = [r for r in results if r.get('test_suite') == 'memory_analysis']

        if not memory_tests:
            return {"error": "No memory analysis tests found"}

        analysis = {
            "test_type": "memory_usage_analysis",
            "num_tests": len(memory_tests),
            "memory_efficiency": {}
        }

        # Extract memory data
        peak_memory_usage = []
        memory_delta = []

        for test in memory_tests:
            for subtest in test.get('tests', []):
                if 'peak_ram_used_gb' in subtest:
                    peak_memory_usage.append(subtest['peak_ram_used_gb'])
                if 'memory_delta_gb' in subtest:
                    memory_delta.append(subtest['memory_delta_gb'])

        if peak_memory_usage:
            analysis["memory_efficiency"]["peak_usage"] = {
                "mean": np.mean(peak_memory_usage),
                "max": np.max(peak_memory_usage),
                "memory_efficient": np.mean(peak_memory_usage) < 4  # Less than 4GB average
            }

        if memory_delta:
            analysis["memory_efficiency"]["memory_leak"] = {
                "mean_delta": np.mean(memory_delta),
                "max_delta": np.max(memory_delta),
                "leak_detected": np.mean(memory_delta) > 0.1  # More than 100MB growth
            }

        return analysis

    def analyze_llm_integration(self, results):
        """Analyze LLM integration performance"""

        llm_tests = [r for r in results if r.get('test_suite') == 'llm_integration']

        if not llm_tests:
            return {"error": "No LLM integration tests found"}

        analysis = {
            "test_type": "llm_integration_analysis",
            "num_tests": len(llm_tests),
            "llm_performance": {}
        }

        # Extract LLM data
        savings_data = []
        throughput_data = []

        for test in llm_tests:
            comparison = test.get('comparison', {})
            if 'average_savings_percent' in comparison:
                savings_data.append(comparison['average_savings_percent'])
            if 'overall_improvement_percent' in comparison:
                throughput_data.append(comparison['overall_improvement_percent'])

        if savings_data:
            analysis["llm_performance"]["savings"] = {
                "mean": np.mean(savings_data),
                "std": np.std(savings_data),
                "real_world_impact": np.mean(savings_data) > 15  # More than 15% real savings
            }

        if throughput_data:
            analysis["llm_performance"]["throughput"] = {
                "mean": np.mean(throughput_data),
                "throughput_improved": np.mean(throughput_data) > 0
            }

        return analysis

    def generate_comprehensive_report(self):
        """Generate comprehensive performance analysis report"""

        print("\n=== GENERATING COMPREHENSIVE PERFORMANCE REPORT ===")

        # Load all results
        all_results = self.load_test_results()

        if not all_results:
            return {"error": "No test results found"}

        report = {
            "report_type": "comprehensive_performance_analysis",
            "generated_at": datetime.now().isoformat(),
            "total_test_files": len(all_results),
            "analyses": {}
        }

        # Run all analyses
        analyses = [
            ("residue_performance", self.analyze_residue_performance),
            ("baseline_comparison", self.analyze_baseline_comparison),
            ("memory_usage", self.analyze_memory_usage),
            ("llm_integration", self.analyze_llm_integration)
        ]

        for analysis_name, analysis_func in analyses:
            try:
                result = analysis_func(all_results)
                if 'error' not in result:
                    report["analyses"][analysis_name] = result
                    print(f"✅ {analysis_name} analysis completed")
                else:
                    print(f"⚠️ {analysis_name} analysis: {result['error']}")
            except Exception as e:
                print(f"❌ Error in {analysis_name} analysis: {e}")

        # Overall assessment
        report["overall_assessment"] = self._generate_overall_assessment(report)

        # Save report
        report_file = self.results_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✅ Comprehensive report saved to: {report_file}")
        return report

    def _generate_overall_assessment(self, report):
        """Generate overall performance assessment"""

        assessment = {
            "performance_score": 0,
            "recommendations": [],
            "strengths": [],
            "areas_for_improvement": []
        }

        analyses = report.get("analyses", {})

        # Performance scoring
        score = 0
        max_score = 0

        # RESIDUE performance
        if "residue_performance" in analyses:
            perf = analyses["residue_performance"]
            throughput = perf["metrics"]["throughput"]["mean"]
            if throughput > 5000:
                score += 3
            elif throughput > 2000:
                score += 2
            elif throughput > 1000:
                score += 1
            max_score += 3
            assessment["strengths"].append("Good computational performance")

        # Savings analysis
        if "baseline_comparison" in analyses:
            savings = analyses["baseline_comparison"]
            avg_savings = savings["savings_analysis"]["savings_percentage"]["mean"]
            if avg_savings > 25:
                score += 3
            elif avg_savings > 15:
                score += 2
            elif avg_savings > 5:
                score += 1
            max_score += 3

            if avg_savings > 10:
                assessment["strengths"].append(f"Significant computational savings ({avg_savings:.1f}%)")

        # Memory efficiency
        if "memory_usage" in analyses:
            mem = analyses["memory_usage"]
            if "memory_efficiency" in mem:
                peak_usage = mem["memory_efficiency"].get("peak_usage", {}).get("mean", 10)
                if peak_usage < 4:
                    score += 2
                elif peak_usage < 8:
                    score += 1
                max_score += 2

                if peak_usage < 6:
                    assessment["strengths"].append("Memory efficient")

        # LLM integration
        if "llm_integration" in analyses:
            llm = analyses["llm_integration"]
            if "llm_performance" in llm:
                savings = llm["llm_performance"].get("savings", {}).get("mean", 0)
                if savings > 15:
                    score += 2
                elif savings > 5:
                    score += 1
                max_score += 2

                if savings > 10:
                    assessment["strengths"].append("Effective LLM integration")

        # Calculate final score
        if max_score > 0:
            assessment["performance_score"] = (score / max_score) * 100

        # Generate recommendations
        if assessment["performance_score"] < 60:
            assessment["recommendations"].append("Run more comprehensive performance tests")
        if assessment["performance_score"] > 80:
            assessment["recommendations"].append("Ready for production deployment")
        if len(assessment["strengths"]) < 2:
            assessment["recommendations"].append("Expand testing coverage")

        return assessment

    def create_visualizations(self, report):
        """Create performance visualization charts"""

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('PROJECT RESIDUE Performance Analysis', fontsize=16)

            analyses = report.get("analyses", {})

            # Plot 1: Inference Performance
            if "residue_performance" in analyses:
                perf = analyses["residue_performance"]
                throughput = perf["metrics"]["throughput"]

                axes[0, 0].bar(['Mean', 'Median', 'Max'],
                              [throughput['mean'], throughput['median'], throughput['max']],
                              color=['skyblue', 'lightgreen', 'salmon'])
                axes[0, 0].set_title('Throughput Performance')
                axes[0, 0].set_ylabel('Tokens/Second')
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Savings Analysis
            if "baseline_comparison" in analyses:
                savings = analyses["baseline_comparison"]
                savings_pct = savings["savings_analysis"]["savings_percentage"]

                axes[0, 1].bar(['Mean', 'Median', 'Max'],
                              [savings_pct['mean'], savings_pct['median'], savings_pct['max']],
                              color=['gold', 'orange', 'red'])
                axes[0, 1].set_title('Computational Savings')
                axes[0, 1].set_ylabel('Savings (%)')
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Memory Usage
            if "memory_usage" in analyses:
                mem = analyses["memory_usage"]
                if "memory_efficiency" in mem and "peak_usage" in mem["memory_efficiency"]:
                    peak = mem["memory_efficiency"]["peak_usage"]
                    axes[1, 0].bar(['Mean Peak Usage'], [peak['mean']],
                                  color='lightcoral')
                    axes[1, 0].set_title('Memory Usage')
                    axes[1, 0].set_ylabel('GB')
                    axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Overall Assessment
            assessment = report.get("overall_assessment", {})
            score = assessment.get("performance_score", 0)

            colors = ['lightgray'] * 5
            colors[int(score // 20)] = 'green'  # Fill bars based on score

            axes[1, 1].bar(['0-20', '20-40', '40-60', '60-80', '80-100'],
                          [20, 20, 20, 20, 20], color=colors)
            axes[1, 1].axhline(y=score, color='red', linestyle='--', linewidth=2,
                              label=f'Current Score: {score:.1f}%')
            axes[1, 1].set_title('Performance Score')
            axes[1, 1].set_ylabel('Score Range')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = self.results_dir / f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Performance visualizations saved to: {plot_file}")
            return str(plot_file)

        except ImportError:
            print("⚠️ Matplotlib/seaborn not available for visualizations")
            return None
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return None

def main():
    """Main analysis function"""

    analyzer = PerformanceAnalyzer()

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()

    # Create visualizations
    viz_file = analyzer.create_visualizations(report)

    print("\n=== ANALYSIS COMPLETE ===")
    print("✅ Comprehensive performance analysis generated")
    if viz_file:
        print("✅ Performance visualizations created")
    print("📊 Check results/ directory for detailed reports")

    return report

if __name__ == "__main__":
    main()
