from pathlib import Path
from tempfile import mkdtemp

from glyc_processing.misc import get_latest_uniprot_release


class ConfigWrapper:
    """
    Wrapper that allows choosing different Configs. Is automatically created as 'glyc_preprocessing.cf' on library import.
    By default the BaseConfig is used, but can be changed with use_config(), e.g. cf.use_config(GalNAcConfig)
    """
    def __init__(self, wrapped_config_class=None):
        if wrapped_config_class is not None:
            self.use_config(wrapped_config_class)

    def use_config(self, wrapped_config_class):
        wrapped_config = wrapped_config_class()
        self.__class__ = type(wrapped_config.__class__.__name__,
                              (self.__class__, wrapped_config.__class__),
                              {})
        self.__dict__ = wrapped_config.__dict__


class FrozenClass:
    """
    Wrapper of BaseConfig that prevents users from misspelling attribute names.
    """
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and key not in dir(self):
            raise TypeError("%r does not have the specified attribute, check your spelling" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


class BaseConfig(FrozenClass):
    """
    Base class of config files for different data formats.
    This class can be extended to create custom Config files if your data format requires different configurations
    A few attributes can be set:
     * TEMP_DIR - Path to the folder to store data-specific temp files (default: New temporary directory w. mkdtemp)
     * UNIPROT_DOWNLOADS_DIR - Path to the folder where release downloads from Uniprot are kept (default: TEMP_DIR)
     * UNIPROT_RELEASE - 'latest' or release in format (YYYY_MM). Use first yearly (2015_01, 2021_01 etc.) for reproducability! (default: 'latest')
     * ALLOWED_AA - A tuple of one-letter amino acids that are glycosylated in the data (default: ('S', 'T'))
     * IGNORE_EXISTING_FILES - Set to true if you want to re-download/process all data-specific temporary files (default: False)
    """
    def __init__(self):
        if getattr(self, '_TEMP_DIR', None) is None:
            self._TEMP_DIR = None
        if not hasattr(self, '_UNIPROT_DOWNLOADS_DIR'):
            self._UNIPROT_DOWNLOADS_DIR = self._TEMP_DIR
        if not hasattr(self, '_UNIPROT_RELEASE'):
            self._UNIPROT_RELEASE = "latest"
        if not hasattr(self, '_ALLOWED_AA'):
            self._ALLOWED_AA = ('S', 'T')
        if not hasattr(self, '_IGNORE_EXISTING_FILES'):
            self._IGNORE_EXISTING_FILES = False

        self._new_TEMP_DIR = True
        self._new_UNIPROT_DOWNLOADS_DIR = True
        self._TRUE_UNIPROT_RELEASE = get_latest_uniprot_release()
        self._freeze()

    @property
    def TEMP_DIR(self):
        if self._new_TEMP_DIR:
            if self._TEMP_DIR is not None:
                self._TEMP_DIR.mkdir(exist_ok=True)
            else:
                self._TEMP_DIR = Path(mkdtemp())
            self._new_TEMP_DIR = False
        return self._TEMP_DIR

    @TEMP_DIR.setter
    def TEMP_DIR(self, value):
        if value is not None:
            self._TEMP_DIR = Path(value)
        else:
            self._TEMP_DIR = None
        self._new_TEMP_DIR = True

    @property
    def UNIPROT_DOWNLOADS_DIR(self):
        if self._new_UNIPROT_DOWNLOADS_DIR:
            if self._UNIPROT_DOWNLOADS_DIR is not None:
                self._UNIPROT_DOWNLOADS_DIR.mkdir(exist_ok=True)
            else:
                self._UNIPROT_DOWNLOADS_DIR = self.TEMP_DIR
            self._new_UNIPROT_DOWNLOADS_DIR = False
        return self._UNIPROT_DOWNLOADS_DIR

    @UNIPROT_DOWNLOADS_DIR.setter
    def UNIPROT_DOWNLOADS_DIR(self, value):
        if value is not None:
            self._UNIPROT_DOWNLOADS_DIR = Path(value)
        else:
            self._UNIPROT_DOWNLOADS_DIR = None
        self._new_UNIPROT_DOWNLOADS_DIR = True

    @property
    def UNIPROT_RELEASE(self):
        return self._UNIPROT_RELEASE

    @UNIPROT_RELEASE.setter
    def UNIPROT_RELEASE(self, value):
        self._UNIPROT_RELEASE = value
        if value == 'latest':
            self._TRUE_UNIPROT_RELEASE = get_latest_uniprot_release()
        else:
            self._TRUE_UNIPROT_RELEASE = value

    @property
    def TRUE_UNIPROT_RELEASE(self):
        return self._TRUE_UNIPROT_RELEASE

    @property
    def ALLOWED_AA(self):
        return self._ALLOWED_AA

    @ALLOWED_AA.setter
    def ALLOWED_AA(self, value):
        self._ALLOWED_AA = value

    @property
    def IGNORE_EXISTING_FILES(self):
        return self._IGNORE_EXISTING_FILES

    @IGNORE_EXISTING_FILES.setter
    def IGNORE_EXISTING_FILES(self, value):
        self._IGNORE_EXISTING_FILES = value

    UNIPROT_LATEST_DELETEDIDS_URL = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/delac_sp.txt'

    @property
    def UNIPROT_RELEASE_SPROT_URL(self):
        return f'https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{self.TRUE_UNIPROT_RELEASE}/knowledgebase/uniprot_sprot-only{self.TRUE_UNIPROT_RELEASE}.tar.gz'

    @property
    def UNIPROT_RELEASE_SPROT_TAR_FILE(self):
        return self.UNIPROT_DOWNLOADS_DIR / f"uniprot_sprot-only{self.TRUE_UNIPROT_RELEASE}.tar.gz"

    UNIPROT_RELEASE_SPROT_TAR_ENTRIES_LOCATION = 'uniprot_sprot.dat.gz'

    UNIPROT_RELEASE_SPROT_TAR_ISOFORMS_LOCATION = 'uniprot_sprot_varsplic.fasta.gz'

    @property
    def UNIPROT_RELEASE_SPROT_BGZ_FILE(self):
        return self.UNIPROT_DOWNLOADS_DIR / f"uniprot_sprot{self.TRUE_UNIPROT_RELEASE}.dat.bgz"

    @property
    def UNIPROT_RELEASE_ISOFORMS_BGZ_FILE(self):
        return self.UNIPROT_DOWNLOADS_DIR / f"uniprot_sprot_varsplic{self.TRUE_UNIPROT_RELEASE}.fasta.bgz"

    @property
    def UNIPROT_RELEASE_DOCS_URL(self):
        return f'https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{self.TRUE_UNIPROT_RELEASE}/knowledgebase/knowledgebase-docs-only{self.TRUE_UNIPROT_RELEASE}.tar.gz'

    @property
    def UNIPROT_RELEASE_DOCS_TAR_FILE(self):
        return self.UNIPROT_DOWNLOADS_DIR / f"knowledgebase-docs-only{self.TRUE_UNIPROT_RELEASE}.tar.gz"

    UNIPROT_RELEASE_DOCS_TAR_DELETEDIDS_LOCATION = 'docs/delac_sp.txt'

    @property
    def UNIPROT_ENTRIES_FILE(self):
        return self.TEMP_DIR / f"entries-{self.TRUE_UNIPROT_RELEASE}.dat.bgz"

    @property
    def UNIPROT_ENTRY_SEQS_FILE(self):
        return self.TEMP_DIR / f"entryseqs-{self.TRUE_UNIPROT_RELEASE}.pkl.gz"

    @property
    def UNIPROT_ISOFORMS_FILE(self):
        return self.TEMP_DIR / f"isoforms-{self.TRUE_UNIPROT_RELEASE}.fasta.bgz"

    @property
    def UNIPROT_ISOFORM_SEQS_FILE(self):
        return self.TEMP_DIR / f"isoformseqs-{self.TRUE_UNIPROT_RELEASE}.pkl.gz"

    @property
    def UNIPROT_IDMAPPING_FILE(self):
        return self.TEMP_DIR / f"idmapping-{self.TRUE_UNIPROT_RELEASE}.pkl.gz"

    @property
    def PEPTIDES_FASTA_FILE(self):
        return self.TEMP_DIR / f"idmapping-{self.TRUE_UNIPROT_RELEASE}.pkl.gz"

    @property
    def CDHIT_PEPTIDES_INPUT_FILE(self):
        return self.TEMP_DIR / 'cdhit_input.fasta'

    @property
    def CDHIT_PEPTIDES_OUTPUT_FILE(self):
        return self.TEMP_DIR / 'cdhit_output.fasta'

    @property
    def CDHIT_PEPTIDES_LOG_FILE(self):
        return self.TEMP_DIR / 'cdhit.log'

    ##### Data field regexes #####

    UNIPROT_REGEX = r"^(?:(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?)$"

    UUID_REGEX = r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[1-5][0-9A-Fa-f]{3}-[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12}$"

    PEPTIDE_REGEX = r"^[XACDEFGHIKLMNPQRSTVWY]+$"
