#!/usr/bin/env python3
"""
Deployment script for LLM Blind Evaluation System

This script handles:
- Environment setup and validation
- Security checks
- Application startup
- Health checks
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Handles deployment of the LLM Blind Evaluation System"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize deployment manager"""
        self.project_root = project_root or Path(__file__).parent.parent
        self.required_dirs = [
            "src", "config", "data", "logs", "data/results", "data/vector_store"
        ]
        self.required_env_vars = [
            "GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "ADMIN_PASSWORD"
        ]
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            logger.error(f"Python 3.9+ required, found {version.major}.{version.minor}")
            return False
        logger.info(f"Python version check passed: {version.major}.{version.minor}")
        return True
    
    def create_directories(self) -> bool:
        """Create required directories"""
        try:
            for dir_path in self.required_dirs:
                full_path = self.project_root / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory ensured: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False
    
    def check_environment_variables(self) -> bool:
        """Check if required environment variables are set"""
        missing_vars = []
        for var in self.required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.info("Please create a .env file with the required variables")
            logger.info("See .env.example for reference")
            return False
        
        # Security check for default admin password
        if os.getenv("ADMIN_PASSWORD") == "changeme":
            logger.warning("⚠️  SECURITY WARNING: Default admin password detected!")
            logger.warning("Please set a secure ADMIN_PASSWORD in your .env file")
            return False
        
        logger.info("Environment variables check passed")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        try:
            logger.info("Installing dependencies...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
            logger.info("Dependencies installed successfully")
            return True
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def run_security_checks(self) -> bool:
        """Run security validation checks"""
        logger.info("Running security checks...")
        
        # Check for .env file in project root (should not exist)
        env_file = self.project_root / ".env"
        if env_file.exists():
            logger.warning("⚠️  .env file found in project root - ensure it's not committed to git")
        
        # Check file permissions on sensitive files
        config_files = [
            ".env.example",
            "config/app_config.yaml",
            "config/llm_config.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                # Check if file is readable by others (security risk)
                stat = file_path.stat()
                if stat.st_mode & 0o044:  # World readable
                    logger.warning(f"⚠️  {config_file} is world-readable - consider restricting permissions")
        
        logger.info("Security checks completed")
        return True
    
    def validate_streamlit_setup(self) -> bool:
        """Validate Streamlit configuration"""
        try:
            # Check if main.py exists
            main_file = self.project_root / "src" / "ui" / "main.py"
            if not main_file.exists():
                logger.error("Main Streamlit app not found at src/ui/main.py")
                return False
            
            # Check start_streamlit.py
            start_file = self.project_root / "start_streamlit.py"
            if not start_file.exists():
                logger.error("Streamlit startup script not found")
                return False
            
            logger.info("Streamlit setup validation passed")
            return True
        except Exception as e:
            logger.error(f"Error validating Streamlit setup: {e}")
            return False
    
    def run_health_check(self) -> bool:
        """Run basic health checks"""
        logger.info("Running health checks...")
        
        try:
            # Import key modules to check for import errors
            sys.path.insert(0, str(self.project_root / "src"))
            
            from llm_providers.provider_manager import ProviderManager
            from security.input_validator import InputValidator
            from rag.pipeline import RAGPipeline
            
            logger.info("Module import checks passed")
            return True
        except ImportError as e:
            logger.error(f"Import error during health check: {e}")
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def start_application(self, host: str = "0.0.0.0", port: int = 8501) -> bool:
        """Start the Streamlit application"""
        try:
            logger.info(f"Starting Streamlit application on {host}:{port}")
            
            # Set environment variables for Streamlit
            os.environ["STREAMLIT_SERVER_ADDRESS"] = host
            os.environ["STREAMLIT_SERVER_PORT"] = str(port)
            os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
            
            # Start Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                "start_streamlit.py",
                "--server.address", host,
                "--server.port", str(port),
                "--browser.gatherUsageStats", "false"
            ]
            
            subprocess.run(cmd, cwd=self.project_root)
            return True
            
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
            return True
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            return False
    
    def deploy(self, start_app: bool = True, host: str = "0.0.0.0", port: int = 8501) -> bool:
        """Run full deployment process"""
        logger.info("Starting deployment process...")
        
        steps = [
            ("Python version check", self.check_python_version),
            ("Directory creation", self.create_directories),
            ("Environment variables check", self.check_environment_variables),
            ("Dependencies installation", self.install_dependencies),
            ("Security checks", self.run_security_checks),
            ("Streamlit setup validation", self.validate_streamlit_setup),
            ("Health checks", self.run_health_check),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Running: {step_name}")
            if not step_func():
                logger.error(f"❌ Failed: {step_name}")
                return False
            logger.info(f"✅ Passed: {step_name}")
        
        logger.info("🎉 Deployment validation completed successfully!")
        
        if start_app:
            logger.info("Starting application...")
            return self.start_application(host, port)
        
        return True


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy LLM Blind Evaluation System")
    parser.add_argument("--no-start", action="store_true", help="Skip starting the application")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8501, help="Port number (default: 8501)")
    parser.add_argument("--check-only", action="store_true", help="Run checks only, don't start")
    
    args = parser.parse_args()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not available, skipping .env file loading")
    
    # Initialize deployment manager
    deployer = DeploymentManager()
    
    # Run deployment
    start_app = not args.no_start and not args.check_only
    success = deployer.deploy(start_app=start_app, host=args.host, port=args.port)
    
    if not success:
        logger.error("❌ Deployment failed!")
        sys.exit(1)
    
    if args.check_only:
        logger.info("✅ All deployment checks passed!")
    elif not start_app:
        logger.info("✅ Deployment ready! Run with --start to launch the application.")


if __name__ == "__main__":
    main() 