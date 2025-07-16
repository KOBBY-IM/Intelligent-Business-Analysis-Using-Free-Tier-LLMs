"""
Secure Logging Utilities

This module provides secure logging functionality that prevents PII leakage
and ensures privacy compliance for the LLM evaluation system.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SecureLogger:
    """Secure logger that prevents PII leakage"""
    
    # Patterns that might contain PII
    PII_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',  # IBAN
        r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card with spaces
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email again
    ]
    
    # Fields that should never be logged
    SENSITIVE_FIELDS = {
        'password', 'api_key', 'secret', 'token', 'key', 'credential',
        'ssn', 'social_security', 'credit_card', 'card_number',
        'phone', 'email', 'address', 'name', 'username', 'user_id',
        'session_id', 'ip_address', 'mac_address'
    }
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize secure logger
        
        Args:
            log_file: Optional log file path
        """
        self.log_file = log_file
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.PII_PATTERNS]
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Configure formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing or masking PII
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Replace PII patterns with placeholders
        for pattern in self.compiled_patterns:
            text = pattern.sub('[REDACTED]', text)
        
        return text
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize dictionary by removing sensitive fields and masking PII
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        if not data:
            return {}
        
        sanitized = {}
        
        for key, value in data.items():
            # Check if key contains sensitive terms
            key_lower = key.lower()
            is_sensitive = any(term in key_lower for term in self.SENSITIVE_FIELDS)
            
            if is_sensitive:
                # Replace sensitive values with placeholder
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, str):
                # Sanitize string values
                sanitized[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                # Sanitize list items
                sanitized[key] = [
                    self.sanitize_text(item) if isinstance(item, str)
                    else self.sanitize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                # Keep other types as is
                sanitized[key] = value
        
        return sanitized
    
    def log_event(self, 
                  event_type: str, 
                  message: str, 
                  data: Optional[Dict[str, Any]] = None,
                  level: str = 'INFO'):
        """
        Log an event securely
        
        Args:
            event_type: Type of event (e.g., 'user_login', 'api_call')
            message: Event message
            data: Optional event data
            level: Log level
        """
        # Sanitize data
        sanitized_data = self.sanitize_dict(data) if data else {}
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': self.sanitize_text(message),
            'data': sanitized_data
        }
        
        # Log based on level
        if level.upper() == 'DEBUG':
            logger.debug(json.dumps(log_entry))
        elif level.upper() == 'INFO':
            logger.info(json.dumps(log_entry))
        elif level.upper() == 'WARNING':
            logger.warning(json.dumps(log_entry))
        elif level.upper() == 'ERROR':
            logger.error(json.dumps(log_entry))
        elif level.upper() == 'CRITICAL':
            logger.critical(json.dumps(log_entry))
        else:
            logger.info(json.dumps(log_entry))
    
    def log_user_action(self, 
                       user_id: str, 
                       action: str, 
                       details: Optional[Dict[str, Any]] = None):
        """
        Log user actions securely
        
        Args:
            user_id: User identifier (will be hashed)
            action: Action performed
            details: Optional action details
        """
        # Hash user ID for privacy
        hashed_user_id = self._hash_identifier(user_id)
        
        # Sanitize details
        sanitized_details = self.sanitize_dict(details) if details else {}
        
        self.log_event(
            event_type='user_action',
            message=f"User {hashed_user_id} performed action: {action}",
            data={
                'user_id_hash': hashed_user_id,
                'action': action,
                'details': sanitized_details
            }
        )
    
    def log_api_call(self, 
                    provider: str, 
                    model: str, 
                    query_length: int,
                    response_time: float,
                    success: bool,
                    error_message: Optional[str] = None):
        """
        Log API calls securely
        
        Args:
            provider: API provider name
            model: Model name
            query_length: Length of query (not the actual query)
            response_time: Response time in seconds
            success: Whether call was successful
            error_message: Error message if failed
        """
        data = {
            'provider': provider,
            'model': model,
            'query_length': query_length,
            'response_time': response_time,
            'success': success
        }
        
        if error_message:
            data['error_message'] = self.sanitize_text(error_message)
        
        self.log_event(
            event_type='api_call',
            message=f"API call to {provider}/{model}",
            data=data
        )
    
    def log_evaluation_submission(self, 
                                 industry: str,
                                 question_id: str,
                                 selected_response: str,
                                 has_comment: bool):
        """
        Log evaluation submissions securely
        
        Args:
            industry: Industry being evaluated
            question_id: Question identifier
            selected_response: Selected response label (A-F)
            has_comment: Whether user provided a comment
        """
        self.log_event(
            event_type='evaluation_submission',
            message=f"Evaluation submitted for {industry} industry",
            data={
                'industry': industry,
                'question_id': question_id,
                'selected_response': selected_response,
                'has_comment': has_comment
            }
        )
    
    def log_security_event(self, 
                          event_type: str, 
                          severity: str,
                          details: Optional[Dict[str, Any]] = None):
        """
        Log security events
        
        Args:
            event_type: Type of security event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            details: Event details
        """
        sanitized_details = self.sanitize_dict(details) if details else {}
        
        self.log_event(
            event_type=f'security_{event_type}',
            message=f"Security event: {event_type} (severity: {severity})",
            data={
                'severity': severity,
                'details': sanitized_details
            },
            level='WARNING' if severity in ['HIGH', 'CRITICAL'] else 'INFO'
        )
    
    def _hash_identifier(self, identifier: str) -> str:
        """
        Hash an identifier for privacy
        
        Args:
            identifier: Identifier to hash
            
        Returns:
            Hashed identifier
        """
        import hashlib
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics
        
        Returns:
            Dictionary with log statistics
        """
        if not self.log_file or not Path(self.log_file).exists():
            return {'total_entries': 0, 'file_size_mb': 0}
        
        log_path = Path(self.log_file)
        file_size_mb = log_path.stat().st_size / (1024 * 1024)
        
        # Count log entries
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                total_entries = len(lines)
        except Exception:
            total_entries = 0
        
        return {
            'total_entries': total_entries,
            'file_size_mb': round(file_size_mb, 2),
            'log_file': str(log_path)
        }


class AuditLogger:
    """Audit logger for compliance and research purposes"""
    
    def __init__(self, audit_file: str = "logs/audit.log"):
        """
        Initialize audit logger
        
        Args:
            audit_file: Audit log file path
        """
        self.audit_file = audit_file
        self.secure_logger = SecureLogger(audit_file)
    
    def log_research_event(self, 
                          event_type: str,
                          participant_id: str,
                          session_id: str,
                          data: Dict[str, Any]):
        """
        Log research events for compliance
        
        Args:
            event_type: Type of research event
            participant_id: Anonymous participant ID
            session_id: Session identifier
            data: Research data
        """
        # Ensure participant ID is hashed
        hashed_participant = self.secure_logger._hash_identifier(participant_id)
        
        sanitized_data = self.secure_logger.sanitize_dict(data)
        
        self.secure_logger.log_event(
            event_type=f'research_{event_type}',
            message=f"Research event: {event_type}",
            data={
                'participant_id_hash': hashed_participant,
                'session_id': session_id,
                'research_data': sanitized_data
            }
        )
    
    def log_consent_event(self, 
                         participant_id: str,
                         consent_given: bool,
                         timestamp: str):
        """
        Log consent events
        
        Args:
            participant_id: Anonymous participant ID
            consent_given: Whether consent was given
            timestamp: Consent timestamp
        """
        hashed_participant = self.secure_logger._hash_identifier(participant_id)
        
        self.secure_logger.log_event(
            event_type='consent_event',
            message=f"Consent {'given' if consent_given else 'withdrawn'}",
            data={
                'participant_id_hash': hashed_participant,
                'consent_given': consent_given,
                'timestamp': timestamp
            }
        ) 