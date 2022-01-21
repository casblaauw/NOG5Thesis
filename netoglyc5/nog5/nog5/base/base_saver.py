from pathlib import Path

from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


class SaverBase:

    def __init__(self, output_path: str):
        """ Check if file should be in write or append mode """
        self.output_path = Path(output_path)
        self.mode = 'a' if self.output_path.exists() else 'w'

    def write(self, identifiers, sequences, data, target, mask, output):
        """ Feed batches to file for writing/appending """
        raise NotImplementedError

    def close(self):
        """ Make sure that file is closed """
        raise NotImplementedError
