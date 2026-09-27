import re

from dataclasses import dataclass
from enum import Enum

# PREFIXES that should be cleaned up before passing into model
PREFIXES = (
    "pick up ",
    "find ",
    "locate ",
    "detect ",
)

class Command(Enum):
    DEFAULT = "default"

commands = [cmd.value for cmd in Command]

@dataclass
class SanitizedInput:
    input: str | None
    is_valid: bool
    command : Command | None = None

class InputSanitizer():
    """
    SanitizeInput returns SanitizedInput object with a boolean indicating whether the input
    is valid and a string indicating the sanitized input, after performing basic validation.
    """
    
    def __init__(
        self,
        input: str | None
    ) -> None:
        self._input: str | None = input

    @staticmethod
    def _is_valid_detection_description(description: str) -> bool:

        if not description:
            return False

        words = description.split()

        if len(words) > 10:
            return False

        if any(len(word) > 20 for word in words):
            return False

        return bool(
            re.fullmatch(
                r"[A-Za-z][A-Za-z\s'-]*",
                description,
            )
        )

    @staticmethod
    def _extract_object_description(command: str) -> str:

        normalised_command = command.strip()

        for prefix in PREFIXES:
            if normalised_command.lower().startswith(prefix):
                return normalised_command[len(prefix):].strip()

        return normalised_command

    @staticmethod
    def _is_valid_command(description: str) -> Command | None:

        if not description:
            return None

        for cmd in commands:
            if description.lower() == cmd:
                return cmd

        return None

    def process_input(self) -> SanitizedInput:
        """Process the user input and return a detection description."""
        # Sanitizing input by stripping white spaces, extracting description and checking valid description
        if self._input is None:
            return SanitizedInput(
                input = None,
                is_valid = False
            )

        command_if_found = self._is_valid_command(self._input)

        if command_if_found:
            return SanitizedInput(
                input=self._input,
                is_valid=True,
                command=command_if_found,
            )

        input_text = self._input.strip()
        input_text = self._extract_object_description(input_text)
        
        return SanitizedInput(
            input = input_text,
            is_valid = self._is_valid_detection_description(input_text)
        )
