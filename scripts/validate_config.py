#!/usr/bin/env python3
"""
Configuration Validation Script

Validates all configuration files for security and completeness
before deployment.
"""

import os
import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from security.input_validator import InputValidator
from config.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_env_file() -> bool:
    """Validate .env file security and completeness"""
    logger.info("Validating environment configuration...")
    
    # Check if .env.example exists
    env_example = Path(".env.example")
    if not env_example.exists():
        logger.error("❌ .env.example file not found")
        return False
    
    # Check if .env exists for deployment
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️  .env file not found - create one from .env.example")
        return False
    
    # Validate required environment variables
    required_vars = [
        "GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "ADMIN_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check for default/weak passwords
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    if admin_password in ["changeme", "admin", "password", "123456"]:
        logger.error("❌ Weak admin password detected - please use a strong password")
        return False
    
    logger.info("✅ Environment configuration valid")
    return True


def validate_yaml_configs() -> bool:
    """Validate YAML configuration files"""
    logger.info("Validating YAML configuration files...")
    
    config_files = [
        "config/app_config.yaml",
        "config/llm_config.yaml", 
        "config/evaluation_config.yaml",
        "config/logging_config.yaml"
    ]
    
    for config_file in config_files:
        file_path = Path(config_file)
        if not file_path.exists():
            logger.error(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            logger.info(f"✅ Valid YAML: {config_file}")
        except yaml.YAMLError as e:
            logger.error(f"❌ Invalid YAML in {config_file}: {e}")
            return False
    
    return True


def validate_model_configurations() -> bool:
    """Validate LLM model configurations"""
    logger.info("Validating LLM model configurations...")
    
    try:
        config_loader = ConfigLoader()
        llm_config = config_loader.load_llm_config()
        
        # Check if providers section exists
        if "providers" not in llm_config:
            logger.error("❌ Missing 'providers' section in LLM config")
            return False
        
        providers = llm_config["providers"]
        
        # Check if all provider sections exist
        required_providers = ["groq", "gemini", "openrouter"]
        for provider in required_providers:
            if provider not in providers:
                logger.error(f"❌ Missing provider configuration: {provider}")
                return False
            
            # Check if models are defined
            if "models" not in providers[provider]:
                logger.error(f"❌ No models defined for provider: {provider}")
                return False
            
            # Check if at least one model exists
            if not providers[provider]["models"]:
                logger.error(f"❌ No models defined for provider: {provider}")
                return False
        
        logger.info("✅ LLM model configurations valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating model configurations: {e}")
        return False


def validate_data_directories() -> bool:
    """Validate data directory structure"""
    logger.info("Validating data directory structure...")
    
    required_dirs = [
        "data",
        "data/results", 
        "data/vector_store",
        "logs"
    ]
    
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if not full_path.exists():
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create directory {dir_path}: {e}")
                return False
        else:
            logger.info(f"✅ Directory exists: {dir_path}")
    
    return True


def validate_security_settings() -> bool:
    """Validate security configurations"""
    logger.info("Validating security settings...")
    
    try:
        validator = InputValidator()
        
        # Test input validation
        test_cases = [
            ("normal text", True),
            ("<script>alert('xss')</script>", False),
            ("javascript:void(0)", False),
        ]
        
        for test_input, expected_safe in test_cases:
            sanitized = validator.sanitize_text(test_input)
            is_safe = sanitized == test_input
            
            if is_safe != expected_safe:
                logger.error(f"❌ Input validation failed for: {test_input}")
                return False
        
        logger.info("✅ Security validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating security settings: {e}")
        return False


def main():
    """Main validation function"""
    logger.info("🔍 Starting configuration validation...")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✅ Environment variables loaded")
    except ImportError:
        logger.warning("⚠️  python-dotenv not available")
    
    # Run validation steps
    validation_steps = [
        ("Environment file validation", validate_env_file),
        ("YAML configuration validation", validate_yaml_configs),
        ("Model configuration validation", validate_model_configurations),
        ("Data directory validation", validate_data_directories),
        ("Security settings validation", validate_security_settings),
    ]
    
    failed_steps = []
    for step_name, step_func in validation_steps:
        logger.info(f"Running: {step_name}")
        try:
            if step_func():
                logger.info(f"✅ {step_name} passed")
            else:
                logger.error(f"❌ {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    # Summary
    if failed_steps:
        logger.error(f"❌ Configuration validation failed! Failed steps: {', '.join(failed_steps)}")
        sys.exit(1)
    else:
        logger.info("🎉 All configuration validation checks passed!")
        logger.info("✅ System ready for deployment!")


if __name__ == "__main__":
    main() 