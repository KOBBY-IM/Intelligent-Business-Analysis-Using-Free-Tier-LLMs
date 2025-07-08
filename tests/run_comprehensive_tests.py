#!/usr/bin/env python3
"""
Comprehensive test runner for the LLM evaluation system.
Runs all tests and generates detailed reports.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ComprehensiveTestRunner:
    """Comprehensive test runner for the entire system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "data" / "test_results"
        self.backup_dir = project_root / "data" / "backups"
        self.reports_dir = project_root / "data" / "reports"

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Test configuration
        self.test_config = {
            "unit_tests": [
                "tests/test_llm_providers.py",
                "tests/test_evaluation_system.py",
                "tests/test_rag_system.py",
            ],
            "integration_tests": ["tests/test_integration.py"],
            "performance_tests": ["tests/test_performance.py"],
            "security_tests": ["tests/test_security.py"],
        }

        # Test results storage
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_duration": 0,
            "coverage_percentage": 0,
            "test_categories": {},
            "errors": [],
            "warnings": [],
        }

    def create_system_backup(self) -> str:
        """Create a comprehensive backup of the working system."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"system_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        try:
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup source code
            src_backup = backup_path / "src"
            if (self.project_root / "src").exists():
                shutil.copytree(
                    self.project_root / "src", src_backup, dirs_exist_ok=True
                )

            # Backup configuration
            config_backup = backup_path / "config"
            if (self.project_root / "config").exists():
                shutil.copytree(
                    self.project_root / "config", config_backup, dirs_exist_ok=True
                )

            # Backup tests
            tests_backup = backup_path / "tests"
            if (self.project_root / "tests").exists():
                shutil.copytree(
                    self.project_root / "tests", tests_backup, dirs_exist_ok=True
                )

            # Backup requirements
            for req_file in [
                "requirements.txt",
                "requirements-dev.txt",
                "pyproject.toml",
            ]:
                req_path = self.project_root / req_file
                if req_path.exists():
                    shutil.copy2(req_path, backup_path / req_file)

            # Backup documentation
            docs_backup = backup_path / "docs"
            if (self.project_root / "docs").exists():
                shutil.copytree(
                    self.project_root / "docs", docs_backup, dirs_exist_ok=True
                )

            # Create backup manifest
            manifest = {
                "backup_timestamp": timestamp,
                "backup_type": "comprehensive_system_backup",
                "backup_contents": [
                    "src/ - Source code",
                    "config/ - Configuration files",
                    "tests/ - Test suite",
                    "requirements.txt - Dependencies",
                    "requirements-dev.txt - Development dependencies",
                    "pyproject.toml - Project configuration",
                    "docs/ - Documentation",
                ],
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "project_root": str(self.project_root),
                },
            }

            with open(backup_path / "backup_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            print(f"✅ System backup created: {backup_path}")
            return str(backup_path)

        except Exception as e:
            print(f"❌ Failed to create system backup: {e}")
            return ""

    def restore_system_from_backup(self, backup_path: str) -> bool:
        """Restore system from a backup."""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                print(f"❌ Backup path does not exist: {backup_path}")
                return False

            # Verify backup manifest
            manifest_path = backup_path / "backup_manifest.json"
            if not manifest_path.exists():
                print(f"❌ Backup manifest not found: {manifest_path}")
                return False

            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            print(f"🔄 Restoring system from backup: {backup_path}")
            print(f"   Backup timestamp: {manifest['backup_timestamp']}")

            # Restore source code
            src_backup = backup_path / "src"
            if src_backup.exists():
                shutil.rmtree(self.project_root / "src", ignore_errors=True)
                shutil.copytree(src_backup, self.project_root / "src")

            # Restore configuration
            config_backup = backup_path / "config"
            if config_backup.exists():
                shutil.rmtree(self.project_root / "config", ignore_errors=True)
                shutil.copytree(config_backup, self.project_root / "config")

            # Restore tests
            tests_backup = backup_path / "tests"
            if tests_backup.exists():
                shutil.rmtree(self.project_root / "tests", ignore_errors=True)
                shutil.copytree(tests_backup, self.project_root / "tests")

            # Restore requirements
            for req_file in [
                "requirements.txt",
                "requirements-dev.txt",
                "pyproject.toml",
            ]:
                req_backup = backup_path / req_file
                if req_backup.exists():
                    shutil.copy2(req_backup, self.project_root / req_file)

            print("✅ System restored successfully from backup")
            return True

        except Exception as e:
            print(f"❌ Failed to restore system from backup: {e}")
            return False

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run all unit tests."""
        print("\n🧪 Running Unit Tests...")
        start_time = time.time()

        results = {
            "category": "unit_tests",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "duration": 0,
        }

        for test_file in self.test_config["unit_tests"]:
            test_path = self.project_root / test_file
            if not test_path.exists():
                print(f"⚠️  Test file not found: {test_file}")
                continue

            print(f"   Running: {test_file}")
            try:
                # Run pytest with coverage
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_path),
                    "-v",
                    "--tb=short",
                    "--cov=src",
                    "--cov-report=term-missing",
                    "--cov-report=html:data/reports/coverage",
                    "--junit-xml=data/test_results/unit_tests.xml",
                ]

                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout
                )

                # Parse results
                if result.returncode == 0:
                    results["passed"] += 1
                    print(f"   ✅ {test_file} - PASSED")
                else:
                    results["failed"] += 1
                    results["errors"].append(
                        {"file": test_file, "error": result.stderr}
                    )
                    print(f"   ❌ {test_file} - FAILED")

                results["tests_run"] += 1

            except subprocess.TimeoutExpired:
                results["failed"] += 1
                results["errors"].append({"file": test_file, "error": "Test timeout"})
                print(f"   ⏰ {test_file} - TIMEOUT")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"file": test_file, "error": str(e)})
                print(f"   ❌ {test_file} - ERROR: {e}")

        results["duration"] = time.time() - start_time
        return results

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("\n🔗 Running Integration Tests...")
        start_time = time.time()

        results = {
            "category": "integration_tests",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "duration": 0,
        }

        for test_file in self.test_config["integration_tests"]:
            test_path = self.project_root / test_file
            if not test_path.exists():
                print(f"⚠️  Test file not found: {test_file}")
                continue

            print(f"   Running: {test_file}")
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_path),
                    "-v",
                    "--tb=short",
                    "--junit-xml=data/test_results/integration_tests.xml",
                ]

                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes timeout for integration tests
                )

                if result.returncode == 0:
                    results["passed"] += 1
                    print(f"   ✅ {test_file} - PASSED")
                else:
                    results["failed"] += 1
                    results["errors"].append(
                        {"file": test_file, "error": result.stderr}
                    )
                    print(f"   ❌ {test_file} - FAILED")

                results["tests_run"] += 1

            except subprocess.TimeoutExpired:
                results["failed"] += 1
                results["errors"].append({"file": test_file, "error": "Test timeout"})
                print(f"   ⏰ {test_file} - TIMEOUT")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"file": test_file, "error": str(e)})
                print(f"   ❌ {test_file} - ERROR: {e}")

        results["duration"] = time.time() - start_time
        return results

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        print("\n⚡ Running Performance Tests...")
        start_time = time.time()

        results = {
            "category": "performance_tests",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "duration": 0,
            "performance_metrics": {},
        }

        # Run performance benchmarks
        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_performance.py",
                "-v",
                "--tb=short",
                "--junit-xml=data/test_results/performance_tests.xml",
            ]

            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                results["passed"] += 1
                print("   ✅ Performance tests - PASSED")
            else:
                results["failed"] += 1
                results["errors"].append(
                    {"file": "performance_tests", "error": result.stderr}
                )
                print("   ❌ Performance tests - FAILED")

            results["tests_run"] += 1

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"file": "performance_tests", "error": str(e)})
            print(f"   ❌ Performance tests - ERROR: {e}")

        results["duration"] = time.time() - start_time
        return results

    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        print("\n🔒 Running Security Tests...")
        start_time = time.time()

        results = {
            "category": "security_tests",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "duration": 0,
            "security_vulnerabilities": [],
        }

        # Run security tests
        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_security.py",
                "-v",
                "--tb=short",
                "--junit-xml=data/test_results/security_tests.xml",
            ]

            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                results["passed"] += 1
                print("   ✅ Security tests - PASSED")
            else:
                results["failed"] += 1
                results["errors"].append(
                    {"file": "security_tests", "error": result.stderr}
                )
                print("   ❌ Security tests - FAILED")

            results["tests_run"] += 1

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"file": "security_tests", "error": str(e)})
            print(f"   ❌ Security tests - ERROR: {e}")

        results["duration"] = time.time() - start_time
        return results

    def generate_coverage_report(self) -> float:
        """Generate coverage report and return coverage percentage."""
        print("\n📊 Generating Coverage Report...")

        try:
            cmd = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--source=src",
                "-m",
                "pytest",
                "tests/",
                "--tb=short",
            ]

            subprocess.run(cmd, cwd=self.project_root, capture_output=True, timeout=600)

            # Generate coverage report
            cmd = [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--show-missing",
                "--format=json",
            ]

            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True
            )

            if result.returncode == 0:
                coverage_data = json.loads(result.stdout)
                total_coverage = coverage_data.get("totals", {}).get(
                    "percent_covered", 0.0
                )
                print(f"   📈 Total Coverage: {total_coverage:.2f}%")
                return total_coverage
            else:
                print("   ⚠️  Could not generate coverage report")
                return 0.0

        except Exception as e:
            print(f"   ❌ Coverage report generation failed: {e}")
            return 0.0

    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        print("\n📋 Generating Test Report...")

        report_path = (
            self.reports_dir
            / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        # Calculate totals
        total_tests = sum(
            cat["tests_run"] for cat in self.test_results["test_categories"].values()
        )
        total_passed = sum(
            cat["passed"] for cat in self.test_results["test_categories"].values()
        )
        total_failed = sum(
            cat["failed"] for cat in self.test_results["test_categories"].values()
        )
        total_skipped = sum(
            cat["skipped"] for cat in self.test_results["test_categories"].values()
        )

        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Evaluation System - Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .category {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
                .error {{ background-color: #ffe6e6; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LLM Evaluation System - Comprehensive Test Report</h1>
                <p><strong>Generated:</strong> {self.test_results["timestamp"]}</p>
                <p><strong>Duration:</strong> {self.test_results["test_duration"]:.2f} seconds</p>
                <p><strong>Coverage:</strong> {self.test_results["coverage_percentage"]:.2f}%</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Passed:</strong> <span class="passed">{total_passed}</span></p>
                <p><strong>Failed:</strong> <span class="failed">{total_failed}</span></p>
                <p><strong>Skipped:</strong> <span class="skipped">{total_skipped}</span></p>
            </div>
        """

        # Add category details
        for category, results in self.test_results["test_categories"].items():
            html_content += f"""
            <div class="category">
                <h3>{category.replace('_', ' ').title()}</h3>
                <p><strong>Tests Run:</strong> {results['tests_run']}</p>
                <p><strong>Passed:</strong> <span class="passed">{results['passed']}</span></p>
                <p><strong>Failed:</strong> <span class="failed">{results['failed']}</span></p>
                <p><strong>Duration:</strong> {results['duration']:.2f} seconds</p>
            """

            if results["errors"]:
                html_content += "<h4>Errors:</h4>"
                for error in results["errors"]:
                    html_content += f"""
                    <div class="error">
                        <strong>{error['file']}:</strong> {error['error']}
                    </div>
                    """

            html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        with open(report_path, "w") as f:
            f.write(html_content)

        print(f"   📄 Report saved: {report_path}")
        return str(report_path)

    def run_all_tests(self) -> bool:
        """Run all tests and generate comprehensive report."""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 50)

        start_time = time.time()

        # Create system backup before testing
        backup_path = self.create_system_backup()

        try:
            # Run all test categories
            unit_results = self.run_unit_tests()
            integration_results = self.run_integration_tests()
            performance_results = self.run_performance_tests()
            security_results = self.run_security_tests()

            # Store results
            self.test_results["test_categories"] = {
                "unit_tests": unit_results,
                "integration_tests": integration_results,
                "performance_tests": performance_results,
                "security_tests": security_results,
            }

            # Generate coverage report
            coverage = self.generate_coverage_report()
            self.test_results["coverage_percentage"] = coverage

            # Calculate totals
            total_tests = sum(
                cat["tests_run"]
                for cat in self.test_results["test_categories"].values()
            )
            total_passed = sum(
                cat["passed"] for cat in self.test_results["test_categories"].values()
            )
            total_failed = sum(
                cat["failed"] for cat in self.test_results["test_categories"].values()
            )

            self.test_results.update(
                {
                    "total_tests": total_tests,
                    "passed_tests": total_passed,
                    "failed_tests": total_failed,
                    "test_duration": time.time() - start_time,
                }
            )

            # Generate report
            report_path = self.generate_test_report()

            # Save results to JSON
            results_file = (
                self.results_dir
                / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(results_file, "w") as f:
                json.dump(self.test_results, f, indent=2)

            # Print summary
            print("\n" + "=" * 50)
            print("📊 TEST SUMMARY")
            print("=" * 50)
            print(f"Total Tests: {total_tests}")
            print(f"Passed: {total_passed}")
            print(f"Failed: {total_failed}")
            print(f"Coverage: {coverage:.2f}%")
            print(f"Duration: {self.test_results['test_duration']:.2f} seconds")
            print(f"Report: {report_path}")
            print(f"Backup: {backup_path}")

            success = total_failed == 0
            if success:
                print("\n✅ All tests passed!")
            else:
                print(f"\n❌ {total_failed} tests failed!")

            return success

        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            return False


def main():
    """Main function to run comprehensive tests."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for LLM evaluation system"
    )
    parser.add_argument(
        "--backup-only", action="store_true", help="Only create system backup"
    )
    parser.add_argument("--restore", type=str, help="Restore system from backup path")
    parser.add_argument(
        "--skip-backup", action="store_true", help="Skip creating backup before tests"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    runner = ComprehensiveTestRunner(project_root)

    if args.restore:
        success = runner.restore_system_from_backup(args.restore)
        sys.exit(0 if success else 1)

    if args.backup_only:
        backup_path = runner.create_system_backup()
        sys.exit(0 if backup_path else 1)

    # Run comprehensive tests
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
