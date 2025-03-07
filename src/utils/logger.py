import os
import json
from datetime import datetime
from pathlib import Path

class ConversationLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_interaction(self, session_id: str, user_message: str, assistant_message: str, metadata: dict = None):
        """
        Log a single interaction between user and assistant.
        
        Args:
            session_id: Unique identifier for the user session
            user_message: The user's message
            assistant_message: The assistant's response
            metadata: Additional metadata about the interaction (optional)
        """
        # Create session directory if it doesn't exist
        session_dir = self.log_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "assistant_message": assistant_message,
            "metadata": metadata or {}
        }
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"interaction_{timestamp}.json"
        
        # Write to file
        log_file = session_dir / filename
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
    def get_session_logs(self, session_id: str):
        """
        Retrieve all logs for a specific session.
        
        Args:
            session_id: The session ID to retrieve logs for
            
        Returns:
            List of log entries sorted by timestamp
        """
        session_dir = self.log_dir / session_id
        if not session_dir.exists():
            return []
            
        logs = []
        for log_file in session_dir.glob("interaction_*.json"):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs.append(json.load(f))
                
        return sorted(logs, key=lambda x: x["timestamp"])
    
    def get_all_sessions(self):
        """
        Get a list of all session IDs.
        
        Returns:
            List of session IDs
        """
        return [d.name for d in self.log_dir.iterdir() if d.is_dir()]
    
    def get_session_summary(self, session_id: str):
        """
        Get a summary of a session's interactions.
        
        Args:
            session_id: The session ID to summarize
            
        Returns:
            Dictionary containing session summary statistics
        """
        logs = self.get_session_logs(session_id)
        if not logs:
            return None
            
        return {
            "session_id": session_id,
            "total_interactions": len(logs),
            "start_time": logs[0]["timestamp"],
            "end_time": logs[-1]["timestamp"],
            "duration_seconds": (datetime.fromisoformat(logs[-1]["timestamp"]) - 
                               datetime.fromisoformat(logs[0]["timestamp"])).total_seconds()
        } 