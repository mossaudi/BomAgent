# config.py - Simplified Configuration Management
"""
Clean configuration management with validation
"""
import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


@dataclass
class SiliconExpertConfig:
    """Silicon Expert API configuration"""
    username: str
    api_key: str
    base_url: str = "https://api.siliconexpert.com/ProductAPI"

    def validate(self) -> List[str]:
        errors = []
        if not self.username:
            errors.append("SILICON_EXPERT_USERNAME is required")
        if not self.api_key:
            errors.append("SILICON_EXPERT_API_KEY is required")
        return errors


@dataclass
class AppConfig:
    """Main application configuration"""
    google_api_key: str
    silicon_expert: SiliconExpertConfig
    debug: bool = False
    max_conversation_history: int = 50
    session_timeout_hours: int = 2

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables"""
        load_dotenv()

        silicon_expert_config = SiliconExpertConfig(
            username=os.getenv("SILICON_EXPERT_USERNAME", ""),
            api_key=os.getenv("SILICON_EXPERT_API_KEY", "")
        )

        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            silicon_expert=silicon_expert_config,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            max_conversation_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "50")),
            session_timeout_hours=int(os.getenv("SESSION_TIMEOUT_HOURS", "2"))
        )

    def validate(self) -> List[str]:
        """Validate all configuration"""
        errors = []

        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")

        errors.extend(self.silicon_expert.validate())

        return errors