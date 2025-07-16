#!/usr/bin/env python3
"""
Feedback Logger Utility

Handles structured logging and secure storage of user feedback data
for the blind evaluation system.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

class FeedbackLogger:
    """Handles structured logging and secure storage of user feedback."""
    
    def __init__(self, results_dir: str = "data/results"):
        """
        Initialize the feedback logger.
        
        Args:
            results_dir: Directory to store feedback results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.results_dir / "user_feedback.json"
        
    def _load_existing_feedback(self) -> List[Dict[str, Any]]:
        """Load existing feedback data from JSON file."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _save_feedback(self, feedback_data: List[Dict[str, Any]]) -> bool:
        """Save feedback data to JSON file with error handling."""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def log_industry_feedback(
        self,
        session_id: str,
        industry: str,
        selected_response: str,
        comment: Optional[str] = None,
        ratings: Optional[Dict[str, int]] = None,
        prompt: Optional[str] = None,
        response_order: Optional[List[str]] = None,
        blind_map: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Log feedback for a single industry evaluation.
        
        Args:
            session_id: Unique session identifier
            industry: Industry being evaluated (retail, finance, healthcare)
            selected_response: Selected response label (A, B, C, D, E, F)
            comment: Optional user comment
            ratings: Optional ratings dictionary
            prompt: The original prompt/question
            response_order: Order of responses as displayed to user
            blind_map: Mapping of blind labels to response IDs
            
        Returns:
            bool: True if successfully logged, False otherwise
        """
        feedback_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "industry": industry,
            "selected_response": selected_response,
            "comment": comment or "",
            "ratings": ratings or {},
            "prompt": prompt or "",
            "response_order": response_order or [],
            "blind_map": blind_map or {},
            "record_type": "industry_feedback"
        }
        
        existing_feedback = self._load_existing_feedback()
        existing_feedback.append(feedback_record)
        
        return self._save_feedback(existing_feedback)
    
    def log_session_start(
        self,
        session_id: str,
        user_consent: bool = True,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log the start of a new evaluation session.
        
        Args:
            session_id: Unique session identifier
            user_consent: Whether user gave consent
            session_metadata: Additional session metadata
            
        Returns:
            bool: True if successfully logged, False otherwise
        """
        session_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": "session_start",
            "user_consent": user_consent,
            "session_metadata": session_metadata or {},
            "record_type": "session_event"
        }
        
        existing_feedback = self._load_existing_feedback()
        existing_feedback.append(session_record)
        
        return self._save_feedback(existing_feedback)
    
    def log_session_completion(
        self,
        session_id: str,
        completed_industries: List[str],
        total_time_seconds: Optional[float] = None,
        completion_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log the completion of an evaluation session.
        
        Args:
            session_id: Unique session identifier
            completed_industries: List of industries that were evaluated
            total_time_seconds: Total time spent on evaluation
            completion_metadata: Additional completion metadata
            
        Returns:
            bool: True if successfully logged, False otherwise
        """
        completion_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": "session_completion",
            "completed_industries": completed_industries,
            "total_time_seconds": total_time_seconds,
            "completion_metadata": completion_metadata or {},
            "record_type": "session_event"
        }
        
        existing_feedback = self._load_existing_feedback()
        existing_feedback.append(completion_record)
        
        return self._save_feedback(existing_feedback)
    
    def get_session_feedback(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all feedback records for a specific session.
        
        Args:
            session_id: Session identifier to retrieve
            
        Returns:
            List of feedback records for the session
        """
        existing_feedback = self._load_existing_feedback()
        return [record for record in existing_feedback if record.get("session_id") == session_id]
    
    def get_industry_feedback(self, industry: str) -> List[Dict[str, Any]]:
        """
        Retrieve all feedback records for a specific industry.
        
        Args:
            industry: Industry to retrieve feedback for
            
        Returns:
            List of feedback records for the industry
        """
        existing_feedback = self._load_existing_feedback()
        return [
            record for record in existing_feedback 
            if record.get("record_type") == "industry_feedback" and record.get("industry") == industry
        ]
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all feedback data.
        
        Returns:
            Dictionary containing feedback summary statistics
        """
        existing_feedback = self._load_existing_feedback()
        
        industry_feedback = [r for r in existing_feedback if r.get("record_type") == "industry_feedback"]
        session_events = [r for r in existing_feedback if r.get("record_type") == "session_event"]
        
        # Count by industry
        industry_counts = {}
        for record in industry_feedback:
            industry = record.get("industry", "unknown")
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        # Count completed sessions
        completed_sessions = [r for r in session_events if r.get("event_type") == "session_completion"]
        
        return {
            "total_feedback_records": len(industry_feedback),
            "total_sessions": len(set(r.get("session_id") for r in existing_feedback)),
            "completed_sessions": len(completed_sessions),
            "industry_counts": industry_counts,
            "last_updated": datetime.now().isoformat()
        }
    
    def export_feedback_csv(self, output_file: Optional[str] = None) -> str:
        """
        Export feedback data to CSV format.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to the exported CSV file
        """
        import pandas as pd
        
        existing_feedback = self._load_existing_feedback()
        industry_feedback = [r for r in existing_feedback if r.get("record_type") == "industry_feedback"]
        
        if not industry_feedback:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(industry_feedback)
        
        # Flatten nested dictionaries
        if 'ratings' in df.columns:
            ratings_df = pd.json_normalize(df['ratings'])
            df = pd.concat([df.drop('ratings', axis=1), ratings_df], axis=1)
        
        # Set output file
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"feedback_export_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        return str(output_file)


def create_session_id() -> str:
    """Create a unique session identifier."""
    return f"session_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"


# Convenience functions for easy access
def log_feedback(
    session_id: str,
    industry: str,
    selected_response: str,
    **kwargs
) -> bool:
    """Convenience function to log feedback."""
    logger = FeedbackLogger()
    return logger.log_industry_feedback(session_id, industry, selected_response, **kwargs)


def get_feedback_summary() -> Dict[str, Any]:
    """Convenience function to get feedback summary."""
    logger = FeedbackLogger()
    return logger.get_feedback_summary()


if __name__ == "__main__":
    # Test the feedback logger
    logger = FeedbackLogger()
    
    # Test session start
    session_id = create_session_id()
    logger.log_session_start(session_id, user_consent=True)
    
    # Test industry feedback
    logger.log_industry_feedback(
        session_id=session_id,
        industry="retail",
        selected_response="A",
        comment="This response was very helpful and practical.",
        ratings={"helpfulness": 5, "accuracy": 4, "clarity": 5}
    )
    
    # Test session completion
    logger.log_session_completion(
        session_id=session_id,
        completed_industries=["retail", "finance", "healthcare"]
    )
    
    # Get summary
    summary = logger.get_feedback_summary()
    print("Feedback Summary:", json.dumps(summary, indent=2)) 