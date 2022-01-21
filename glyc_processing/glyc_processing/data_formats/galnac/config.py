from glyc_processing.data_formats.common.config import BaseConfig


class GalNAcConfig(BaseConfig):
    def __init__(self):
        self._ALLOWED_AA = ('S', 'T', 'Y')
        super().__init__()

    ##### Data field regexes #####

    SITE_COMPOSITION_REGEX = r"^(?:(?:[1-9][0-9]*x.+)+)$"

    QUANTIFICATION_REGEX = r"^(?:potential_ko|potential_wt|rejected_ratio|rejected_singlet_light|rejected_singlet_medium)$"

    QUANTIFICATION_CHANNELS_REGEX = r"^(?:ko:wt|L|M|M:L)$"

    SITE_AMBIGUITY_REGEX = r"^(?:inferred|missing_site_coverage|delta_cn_filter)$"

    QUANTIFICATION_CONFIDENCE_REGEX = r"^(?:high|low|low_sn)$"

    COMPOSITION_REGEX = r"^(?:(?:(?:[1-9][0-9]*x.+)+;)*(?:[1-9][0-9]*x.+)+)$"

    ACTIVATION_REGEX = r"^(?:CID|HCD|ETD)$"

    SOURCE_REGEX = r"^(?:(?P<origin>[a-z]*)_)?(?P<lectin>[a-z]*)(?:_(?P<proteases>(?:(?:[a-z]+),)*(?:[a-z]*)))?$"

    ORIGIN_LECTIN_REGEX = r"^(?:(?:sec|tcl|tissue)_(?:vva|pna))$"

    LECTIN_PROTEASES_REGEX = r"^(?:(?:vva|pna)_(?:(?:try|chy|gluc),)*(?:try|chy|gluc))$"
