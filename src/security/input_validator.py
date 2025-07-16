"""
Input Validation and Sanitization

This module provides utilities for validating and sanitizing user inputs
to prevent injection attacks and ensure data integrity.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class InputValidator:
    """Input validation and sanitization utilities"""
    
    # Dangerous patterns to filter out
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'data:',  # Data protocol
        r'vbscript:',  # VBScript protocol
        r'on\w+\s*=',  # Event handlers
        r'<iframe.*?>',  # Iframe tags
        r'<object.*?>',  # Object tags
        r'<embed.*?>',  # Embed tags
        r'<form.*?>',  # Form tags
        r'<input.*?>',  # Input tags
        r'<textarea.*?>',  # Textarea tags
        r'<select.*?>',  # Select tags
        r'<button.*?>',  # Button tags
        r'<link.*?>',  # Link tags
        r'<meta.*?>',  # Meta tags
        r'<style.*?>.*?</style>',  # Style tags
        r'<link.*?>',  # Link tags
        r'<base.*?>',  # Base tags
        r'<bgsound.*?>',  # Bgsound tags
        r'<marquee.*?>',  # Marquee tags
        r'<applet.*?>',  # Applet tags
        r'<param.*?>',  # Param tags
        r'<source.*?>',  # Source tags
        r'<track.*?>',  # Track tags
        r'<video.*?>',  # Video tags
        r'<audio.*?>',  # Audio tags
        r'<canvas.*?>',  # Canvas tags
        r'<svg.*?>',  # SVG tags
        r'<math.*?>',  # Math tags
        r'<command.*?>',  # Command tags
        r'<details.*?>',  # Details tags
        r'<dialog.*?>',  # Dialog tags
        r'<menu.*?>',  # Menu tags
        r'<menuitem.*?>',  # Menuitem tags
        r'<output.*?>',  # Output tags
        r'<progress.*?>',  # Progress tags
        r'<ruby.*?>',  # Ruby tags
        r'<rt.*?>',  # Rt tags
        r'<rp.*?>',  # Rp tags
        r'<wbr.*?>',  # Wbr tags
        r'<area.*?>',  # Area tags
        r'<map.*?>',  # Map tags
        r'<picture.*?>',  # Picture tags
        r'<figcaption.*?>',  # Figcaption tags
        r'<figure.*?>',  # Figure tags
        r'<main.*?>',  # Main tags
        r'<nav.*?>',  # Nav tags
        r'<section.*?>',  # Section tags
        r'<article.*?>',  # Article tags
        r'<aside.*?>',  # Aside tags
        r'<header.*?>',  # Header tags
        r'<footer.*?>',  # Footer tags
        r'<address.*?>',  # Address tags
        r'<blockquote.*?>',  # Blockquote tags
        r'<dd.*?>',  # Dd tags
        r'<dl.*?>',  # Dl tags
        r'<dt.*?>',  # Dt tags
        r'<fieldset.*?>',  # Fieldset tags
        r'<legend.*?>',  # Legend tags
        r'<optgroup.*?>',  # Optgroup tags
        r'<option.*?>',  # Option tags
        r'<tbody.*?>',  # Tbody tags
        r'<td.*?>',  # Td tags
        r'<tfoot.*?>',  # Tfoot tags
        r'<th.*?>',  # Th tags
        r'<thead.*?>',  # Thead tags
        r'<tr.*?>',  # Tr tags
        r'<col.*?>',  # Col tags
        r'<colgroup.*?>',  # Colgroup tags
        r'<caption.*?>',  # Caption tags
        r'<table.*?>',  # Table tags
        r'<h[1-6].*?>',  # Heading tags
        r'<p.*?>',  # Paragraph tags
        r'<br.*?>',  # Break tags
        r'<hr.*?>',  # Horizontal rule tags
        r'<div.*?>',  # Div tags
        r'<span.*?>',  # Span tags
        r'<strong.*?>',  # Strong tags
        r'<em.*?>',  # Emphasis tags
        r'<b.*?>',  # Bold tags
        r'<i.*?>',  # Italic tags
        r'<u.*?>',  # Underline tags
        r'<s.*?>',  # Strikethrough tags
        r'<del.*?>',  # Delete tags
        r'<ins.*?>',  # Insert tags
        r'<mark.*?>',  # Mark tags
        r'<small.*?>',  # Small tags
        r'<sub.*?>',  # Subscript tags
        r'<sup.*?>',  # Superscript tags
        r'<code.*?>',  # Code tags
        r'<pre.*?>',  # Pre tags
        r'<kbd.*?>',  # Keyboard tags
        r'<samp.*?>',  # Sample tags
        r'<var.*?>',  # Variable tags
        r'<cite.*?>',  # Citation tags
        r'<q.*?>',  # Quote tags
        r'<abbr.*?>',  # Abbreviation tags
        r'<acronym.*?>',  # Acronym tags
        r'<dfn.*?>',  # Definition tags
        r'<time.*?>',  # Time tags
        r'<data.*?>',  # Data tags
        r'<meter.*?>',  # Meter tags
        r'<keygen.*?>',  # Keygen tags
        r'<isindex.*?>',  # Isindex tags
        r'<listing.*?>',  # Listing tags
        r'<plaintext.*?>',  # Plaintext tags
        r'<xmp.*?>',  # Xmp tags
        r'<nextid.*?>',  # Nextid tags
        r'<noembed.*?>',  # Noembed tags
        r'<noframes.*?>',  # Noframes tags
        r'<noscript.*?>',  # Noscript tags
        r'<nobr.*?>',  # Nobr tags
        r'<noindex.*?>',  # Noindex tags
        r'<noprint.*?>',  # Noprint tags
        r'<noreferrer.*?>',  # Noreferrer tags
        r'<nospellcheck.*?>',  # Nospellcheck tags
        r'<notranslate.*?>',  # Notranslate tags
        r'<nowrap.*?>',  # Nowrap tags
        r'<noshade.*?>',  # Noshade tags
        r'<nohref.*?>',  # Nohref tags
        r'<noresize.*?>',  # Noresize tags
        r'<noscroll.*?>',  # Noscroll tags
        r'<noborder.*?>',  # Noborder tags
        r'<nofocus.*?>',  # Nofocus tags
        r'<nohighlight.*?>',  # Nohighlight tags
        r'<nohistory.*?>',  # Nohistory tags
        r'<nopopup.*?>',  # Nopopup tags
        r'<noredirect.*?>',  # Noredirect tags
        r'<norefresh.*?>',  # Norefresh tags
        r'<noreload.*?>',  # Noreload tags
        r'<norepeat.*?>',  # Norepeat tags
        r'<norestore.*?>',  # Norestore tags
        r'<noreturn.*?>',  # Noreturn tags
        r'<norevert.*?>',  # Norevert tags
        r'<norewind.*?>',  # Norewind tags
        r'<norewrite.*?>',  # Norewrite tags
        r'<noscroll.*?>',  # Noscroll tags
        r'<noselect.*?>',  # Noselect tags
        r'<nosort.*?>',  # Nosort tags
        r'<nosubmit.*?>',  # Nosubmit tags
        r'<notab.*?>',  # Notab tags
        r'<notarget.*?>',  # Notarget tags
        r'<notext.*?>',  # Notext tags
        r'<notitle.*?>',  # Notitle tags
        r'<notooltip.*?>',  # Notooltip tags
        r'<notrack.*?>',  # Notrack tags
        r'<notransform.*?>',  # Notransform tags
        r'<notransition.*?>',  # Notransition tags
        r'<notype.*?>',  # Notype tags
        r'<noundo.*?>',  # Noundo tags
        r'<noupdate.*?>',  # Noupdate tags
        r'<novalidate.*?>',  # Novalidate tags
        r'<novisibility.*?>',  # Novisibility tags
        r'<nowait.*?>',  # Nowait tags
        r'<nowarn.*?>',  # Nowarn tags
        r'<nowrap.*?>',  # Nowrap tags
        r'<nowrite.*?>',  # Nowrite tags
        r'<nozoom.*?>',  # Nozoom tags
    ]
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.txt', '.csv', '.json', '.yaml', '.yml', '.md'}
    
    # Maximum input lengths
    MAX_QUERY_LENGTH = 5000
    MAX_COMMENT_LENGTH = 1000
    MAX_FILE_SIZE_MB = 10
    
    def __init__(self):
        """Initialize the input validator"""
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                 for pattern in self.DANGEROUS_PATTERNS]
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input by removing dangerous patterns
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Remove dangerous patterns
        for pattern in self.compiled_patterns:
            text = pattern.sub('', text)
        
        # Remove any remaining HTML-like content
        text = re.sub(r'<[^>]*>', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def validate_query(self, query: str) -> tuple[bool, str]:
        """
        Validate a user query
        
        Args:
            query: Query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Query cannot be empty"
        
        if len(query) > self.MAX_QUERY_LENGTH:
            return False, f"Query too long (max {self.MAX_QUERY_LENGTH} characters)"
        
        # Check for dangerous content
        sanitized = self.sanitize_text(query)
        if sanitized != query:
            return False, "Query contains potentially dangerous content"
        
        return True, ""
    
    def validate_comment(self, comment: str) -> tuple[bool, str]:
        """
        Validate a user comment
        
        Args:
            comment: Comment to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not comment:
            return True, ""  # Comments can be empty
        
        if len(comment) > self.MAX_COMMENT_LENGTH:
            return False, f"Comment too long (max {self.MAX_COMMENT_LENGTH} characters)"
        
        # Check for dangerous content
        sanitized = self.sanitize_text(comment)
        if sanitized != comment:
            return False, "Comment contains potentially dangerous content"
        
        return True, ""
    
    def validate_file_upload(self, file_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Validate a file upload
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        # Check file extension
        if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            return False, f"File type not allowed. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
        
        # Check file size
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_FILE_SIZE_MB:
                return False, f"File too large (max {self.MAX_FILE_SIZE_MB}MB)"
        
        return True, ""
    
    def validate_industry(self, industry: str) -> tuple[bool, str]:
        """
        Validate industry selection
        
        Args:
            industry: Industry to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        allowed_industries = {'retail', 'finance', 'healthcare'}
        
        if not industry:
            return False, "Industry cannot be empty"
        
        if industry.lower() not in allowed_industries:
            return False, f"Invalid industry. Allowed: {', '.join(allowed_industries)}"
        
        return True, ""
    
    def validate_response_selection(self, selection: str) -> tuple[bool, str]:
        """
        Validate response selection (A-F)
        
        Args:
            selection: Selected response label
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        allowed_selections = {'A', 'B', 'C', 'D', 'E', 'F'}
        
        if not selection:
            return False, "Please select a response"
        
        if selection.upper() not in allowed_selections:
            return False, f"Invalid selection. Allowed: {', '.join(allowed_selections)}"
        
        return True, ""
    
    def validate_session_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate session data
        
        Args:
            data: Session data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['session_id', 'step']
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate session ID format
        session_id = data.get('session_id', '')
        if not re.match(r'^session_\d+$', session_id):
            return False, "Invalid session ID format"
        
        # Validate step
        step = data.get('step', 0)
        if not isinstance(step, int) or step < 0 or step > 3:
            return False, "Invalid step value"
        
        return True, ""
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize all string values in a dictionary
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self.sanitize_text(item) if isinstance(item, str) else item 
                                for item in value]
            else:
                sanitized[key] = value
        
        return sanitized 