import math
from itertools import chain
from typing import Callable, List, Union, Tuple, Generator

from glyc_processing.misc import cast_to_int_nan_none, na_to_none


class AnnotationError(Exception):
    pass


class SiteError(AnnotationError):
    pass


class PeptideError(AnnotationError):
    pass


class ProteinError(AnnotationError):
    pass


class Site:
    """
    Represents a single site or unclear site range in an MS peptide spectrum.
    Should not be created directly, instead use ProteinSet.
    """
    def __init__(self, protein_id, peptide_id, seq_sites, single_site_or_unclear_start,
                 unclear_site_end=None, site_annotations=None, index_start=1, end_exclusive=False):
        self.peptide_id = peptide_id
        self.site_annotations = site_annotations

        idx_single_site_or_unclear_start = single_site_or_unclear_start - index_start
        if seq_sites[idx_single_site_or_unclear_start] is None:
            raise SiteError(
                f"protein_id={protein_id} self.peptide_id={self.peptide_id}: idx_single_site_or_unclear_start={idx_single_site_or_unclear_start} is not an allowed AA")

        if unclear_site_end is not None:
            idx_unclear_site_end = unclear_site_end - index_start - int(end_exclusive)
            if seq_sites[idx_unclear_site_end] is None:
                raise SiteError(
                    f"protein_id={protein_id} peptide_id={peptide_id}: idx_unclear_site_end={idx_unclear_site_end} is not an allowed AA")
        else:
            idx_unclear_site_end = idx_single_site_or_unclear_start

        self.possible_site_indices = [idx for idx in range(idx_single_site_or_unclear_start, idx_unclear_site_end + 1)
                                      if seq_sites[idx] is not None]
        self.count = len(self.possible_site_indices)
        if self.count == 0:
            raise SiteError(f"protein_id={protein_id} peptide_id={peptide_id}: Site is empty")


class Peptide:
    """
    Represents a MS peptide spectrum.
    Should not be created directly, instead use ProteinSet.
    """
    def __init__(self, protein_id, protein_length, peptide_id, peptide_start=None, peptide_end=None,
                 peptide_annotations=None, index_start=1, end_exclusive=False):
        self.peptide_id = peptide_id
        self.peptide_annotations = peptide_annotations
        self.sites = []

        if peptide_id is None:
            raise PeptideError(f"protein_id={protein_id} peptide_id={peptide_id}: Peptide must have a peptide_id")

        if (peptide_start is None) != (peptide_end is None):
            raise PeptideError(
                f"protein_id={protein_id} peptide_id={peptide_id}: Peptide must have either both or neither of these: [peptide_start, peptide_end]")

        if peptide_start is not None and peptide_end is not None:
            self.peptide_start_idx = peptide_start - index_start
            self.peptide_end_idx = peptide_end - index_start - int(end_exclusive)
            if self.peptide_start_idx < 0:
                raise PeptideError(
                    f"protein_id={protein_id} peptide_id={peptide_id}: self.peptide_start_idx={self.peptide_start_idx} is less than 0")
            if self.peptide_end_idx > (protein_length - 1):
                raise PeptideError(
                    f"protein_id={protein_id} peptide_id={peptide_id}: self.peptide_end_idx={self.peptide_end_idx} is more than protein_length-1={protein_length - 1}")
        else:
            self.peptide_start_idx = None
            self.peptide_end_idx = None

    def __eq__(self, other):
        if not isinstance(other, Peptide):
            return NotImplemented

        return (
                self.peptide_id == other.peptide_id and
                self.peptide_start_idx == other.peptide_start_idx and
                self.peptide_end_idx == other.peptide_end_idx and
                self.peptide_annotations == other.peptide_annotations
        )


class Protein:
    """
    Represents a protein sequence which MS peptide data was mapped to.
    Should not be created directly, instead use ProteinSet.
    """
    def __init__(self, protein_set, protein_id, protein_seq, allowed_aa, scoring_function, protein_annotations=None):
        self.protein_set = protein_set
        self.protein_id = protein_id
        self.protein_seq = protein_seq
        self.protein_annotations = protein_annotations
        self.peptides = {}
        self.sites = []
        self.seq_sites = [[] if aa in allowed_aa else None for aa in self.protein_seq]
        self.seq_idx_seen_count = [0 for _ in range(len(self.protein_seq))]

        if protein_id is None:
            raise ProteinError(f"protein_id={protein_id}: Must have a protein_id")

        if protein_seq is None or not isinstance(protein_seq, (str, list, tuple)) or len(protein_seq) == 0:
            raise ProteinError(f"protein_id={protein_id}: Must have a non-empty string, list or tuple protein_seq")

        if allowed_aa is None or not isinstance(allowed_aa, (tuple, list)) or len(allowed_aa) == 0:
            raise ProteinError(f"protein_id={protein_id}: Must have a non-empty tuple or list allowed_aa")

        if scoring_function is None:
            raise ProteinError(f"protein_id={protein_id}: Must have a scoring_function")

        if not any(aa in protein_seq for aa in allowed_aa):
            raise ProteinError(f"protein_id={protein_id}: protein_seq has none of the allowed_aa")

    def _add(self, peptide_id, peptide_start=None, peptide_end=None, peptide_annotations=None,
             single_site_or_unclear_start=None, unclear_site_end=None, site_annotations=None):
        has_peptide_pos = False
        has_site_pos = False

        # Add Peptide if it doesn't exist already
        peptide = Peptide(self.protein_id, len(self.protein_seq), peptide_id, peptide_start, peptide_end,
                          peptide_annotations, self.protein_set.index_start, self.protein_set.end_exclusive)
        existing_peptide = self.peptides.get(peptide_id)
        if existing_peptide is not None:
            if peptide != existing_peptide:
                raise PeptideError(
                    f"self.protein_id={self.protein_id} peptide_id={peptide_id}: Inconsistent data for peptide_id")
            peptide = existing_peptide
        else:
            self.peptides[peptide_id] = peptide
            # If Peptide position is included (positive & negative data), we count each seen position in it
            if peptide.peptide_start_idx is not None and peptide.peptide_end_idx is not None:
                for idx in range(peptide.peptide_start_idx, peptide.peptide_end_idx + 1):
                    self.seq_idx_seen_count[idx] += 1

        if peptide.peptide_start_idx is not None and peptide.peptide_end_idx is not None:
            has_peptide_pos = True

        # Add Site if site position is included
        if single_site_or_unclear_start is not None:
            has_site_pos = True
            site = Site(self.protein_id, peptide_id, self.seq_sites, single_site_or_unclear_start,
                        unclear_site_end, site_annotations, self.protein_set.index_start, self.protein_set.end_exclusive)
            peptide.sites.append(site)
            self.sites.append(site)
            for possible_site_idx in site.possible_site_indices:
                self.seq_sites[possible_site_idx].append(site)

            # If Peptide position is not included we count each possible Site position instead
            if not has_peptide_pos:
                for idx in range(site.possible_site_indices[0], site.possible_site_indices[-1] + 1):
                    self.seq_idx_seen_count[idx] += 1

        # If neither Peptide nor Site position is included, an error in the data pre-processing must have occurred
        if not has_peptide_pos and not has_site_pos:
            raise PeptideError(
                f"self.protein_id={self.protein_id} peptide_id={peptide_id}: Has neither peptide nor site position")

    def get_glycosylation_labels(self, start: int = None, end: int = None) -> List[float]:
        """
        Get glycosylation labels for each amino acid in the whole protein sequence or a window.
        :param start: Specify start index of sequence window
        :param end:  Specify end index of sequence window (exclusive)
        :return:
            List of glycosylation labels for each amino acid.
            Labels:
                -1.0 for amino acids that cannot be glycosylated or have not been seen (see cf.ALLOWED_AA for which amino acids can be)
                0.0 to 1.0 for seen glycosylatable amino acids according to scoring function (see ProteinSet.scoring_function)
        """
        labels = [-1.0 for _ in range(len(self.protein_seq[start:end]))]
        for possible_site_idx, (sites, seen_count) in enumerate(zip(self.seq_sites[start:end], self.seq_idx_seen_count[start:end])):
            if sites is not None and seen_count > 0:
                labels[possible_site_idx] = self.protein_set.scoring_function(sites, seen_count)
        return labels

    def get_glycosylation_regions(self, window_size: int = 15, label_fill: int = 0) -> List[int]:
        """
        Returns binary labeled sequence (0/1), annotating regions that have been seen to be glycosylated at some point
        """
        # If sites contain any site objects (which means definite or uncertain glycosylation was found), set window around site to 1
        labels = [label_fill for _ in range(len(self.protein_seq))]
        window_wing = window_size//2
        for possible_site_idx, sites in enumerate(self.seq_sites):
            if sites is not None and len(sites) > 0:
                start_idx = max(possible_site_idx-window_wing, 0)
                end_idx = min(possible_site_idx+window_wing+1, len(self.protein_seq))
                labels[start_idx:end_idx] = [1]*(end_idx - start_idx)
        return labels

    def get_enhanced_glycosylation_regions(self, cutoff = 0.75, window_size: int = 15, label_fill: int = 0) -> List[int]:
        """
        Returns labeled sequence (0/1/2), annotating regions with some glycosylation (1) or core glycosylation (2)
        """
        labels = [label_fill for _ in range(len(self.protein_seq))]
        window_wing = window_size//2
        # If sites contain any site objects (which means definite or uncertain glycosylation was found), set window around site to 1
        for possible_site_idx, sites in enumerate(self.seq_sites):
            if sites is not None and len(sites) > 0:
                start_idx = max(possible_site_idx-window_wing, 0)
                end_idx = min(possible_site_idx+window_wing+1, len(self.protein_seq))
                labels[start_idx:end_idx] = [1]*(end_idx - start_idx)
        # If site scores are equal/above cutoff, set window around site to 2
        for possible_site_idx, (sites, seen_count) in enumerate(zip(self.seq_sites, self.seq_idx_site_seen_count)):
            if sites is not None and len(sites) > 0 and seen_count > 0:
                score = len(sites)/seen_count
                if score >= cutoff:
                    start_idx = max(possible_site_idx-window_wing, 0)
                    end_idx = min(possible_site_idx+window_wing+1, len(self.protein_seq))
                    labels[start_idx:end_idx] = [2]*(end_idx - start_idx)
        return labels
    
    def get_seen_regions(self, window_size: int = 15, label_fill: int = 0) -> List[int]:
        """
        Returns binary labeled sequence (0/1), annotating regions that have data (positive or negative)
        """
        return [1 if seen_count+glyco_region > 0 else 0 for seen_count, glyco_region in zip(self.seq_idx_seen_count, self.get_glycosylation_regions(window_size=window_size, label_fill=label_fill))]

    def get_all_site_windows(self, aa_before_site: int = 15, aa_after_site: int = 15, seq_padding: str = 'X',
                             labels_padding : float = -1.0, label_type: str = 'float', label_int_threshold: float = 0.5,
                             include_unseen_sites: bool = False, start: int = None, end: int = None
                             ) -> Generator[Tuple[List, Union[List[float], float, int]], None, None]:
        """
        Yields sequence windows and labels for all possible glycosylation sites, with padding if needed.
        This method creates data for models with fixed input size.
        :param aa_before_site: Amino acids that should be included before each site.
        :param aa_after_site: Amino acids that should be included after each site.
        :param seq_padding: Sequence padding character for sites at the edge of the protein.
        :param labels_padding: Label padding character for sites at the edge of the protein. Is not relevant for label_type 'int' or 'float'
        :param label_type: Type of label for each site:
                   'list': A score for each amino acid in window (see get_glycosylation_labels())
                   'float': The site score (see ProteinSet.scoring_function)
                   'int': A integer 0 or 1 depending on whether the site score is at least as high as label_int_threshold
        :param label_int_threshold: Threshold when label_type='int'
        :param include_unseen_sites: If true, possible glycosylation sites not seen in the data are also yielded
        :param start: Specify start index of sequence to yield from
        :param end: Specify end index of sequence to yield from (exclusive)
        :return: Generator that yields site windows
        """
        if label_int_threshold <= 0 or label_int_threshold >= 1:
            raise KeyError("label_int_threshold must be a number between 0 and 1 if label_type='int'")

        labels = self.get_glycosylation_labels()

        padded_seq = [aa for aa in chain((seq_padding for _ in range(aa_before_site)), self.protein_seq,
                                         (seq_padding for _ in range(aa_after_site)))]
        padded_labels = [label for label in chain((labels_padding for _ in range(aa_before_site)), labels,
                                                  (labels_padding for _ in range(aa_after_site)))]

        for possible_site_idx, sites in enumerate(self.seq_sites[start:end]):
            if sites is not None and (include_unseen_sites or self.seq_idx_seen_count[possible_site_idx] > 0):
                seq_window = padded_seq[possible_site_idx:possible_site_idx + aa_before_site + 1 + aa_after_site]
                if label_type == 'list':
                    labels_window = padded_labels[possible_site_idx:possible_site_idx + aa_before_site + 1 + aa_after_site]
                elif label_type == 'float':
                    labels_window = padded_labels[possible_site_idx + aa_before_site]
                elif label_type == 'int':
                    labels_window = 1 if padded_labels[possible_site_idx + aa_before_site] >= label_int_threshold else 0
                else:
                    raise KeyError("label_type must be one of 'list', 'float' or 'int'")
                yield seq_window, labels_window

    def __eq__(self, other):
        if not isinstance(other, Protein):
            return NotImplemented

        return (
                self.protein_id == other.protein_id and
                self.protein_seq == other.protein_seq and
                self.protein_annotations == other.protein_annotations
        )


class ProteinSet(dict):
    """
    A set of proteins with glycosylation peptide data, to which new Protein/Peptide/Site data can be added.
    When reading from, can be treated as a dictionary with protein IDs as keys and Proteins as values.
    """
    def __init__(self, allowed_aa: tuple, index_start: int = 1, end_exclusive: bool = False,
                 scoring_function: Callable = None):
        """
        Create a new empty ProteinSet, and set options to be used when ingesting data.
        :param allowed_aa: The glycosylated amino acids in the data. Changes only affect new Proteins/Peptides/Sites.
        :param index_start: 0 or 1 depending on which positions start from. Changes only affect new Peptides/Sites.
        :param end_exclusive: whether the end-position of a site/peptide absolute position range is included or not. Changes only affect new Peptides/Sites.
        :param scoring_function: position scoring function, see mean_counts_score for required function parameters
        """
        super().__init__()
        self.version = "1.0.0"  # Update each time you change any code in this module. Use semantic versioning!
        self.allowed_aa = na_to_none(allowed_aa)
        self.index_start = cast_to_int_nan_none(index_start, False)
        self.end_exclusive = na_to_none(end_exclusive)
        self.scoring_function = na_to_none(scoring_function)
        if self.scoring_function is None:
            self.scoring_function = mean_counts_score

    def _add_get_protein(self, protein_id, protein_seq, protein_annotations):
        protein_id = na_to_none(protein_id)
        protein = Protein(self, protein_id, na_to_none(protein_seq), self.allowed_aa, self.scoring_function,
                          na_to_none(protein_annotations))
        existing_protein = self.get(protein_id)
        if existing_protein is not None:
            if protein != existing_protein:
                raise ProteinError(f"protein_id={protein_id}: Inconsistent data for protein_id")
            protein = existing_protein
        else:
            self[protein_id] = protein
        return protein

    def add(self, protein_id: str, protein_seq: Union[str, list, tuple], peptide_id: str,
            single_site_or_unclear_start: int = None, unclear_site_end: int = None, protein_annotations=None,
            peptide_start: int = None, peptide_end: int = None, peptide_annotations=None, site_annotations=None):
        """
        Main function for adding glycosylation data in a tabular format.
        :param protein_id: A unique ID for the protein sequence, e.g. a Uniprot isoform accession.
        :param protein_seq: The protein sequence.
        :param peptide_id: A unique ID for the peptide, e.g. a UUID. If data has no peptide info, you should create one per site.
        :param single_site_or_unclear_start: Absolute position of site in protein sequence or start of range if site is unclear. Can be left out if only peptide info is needed.
        :param unclear_site_end: Absolute end of range if site position is unclear. Can be left out if only peptide info is needed.
        :param protein_annotations: Any additional information for the Protein.
        :param peptide_start: Absolute start position of peptide in protein sequence if peptide data is available.
        :param peptide_end: Absolute end position of peptide in protein sequence if peptide data is available.
        :param peptide_annotations: Any additional information for the Peptide.
        :param site_annotations: Any additional information for the Site.
        :return: Adds/Updates Protein, Peptide & Site information in set.
        """
        peptide_id = na_to_none(peptide_id)
        protein = self._add_get_protein(protein_id, protein_seq, protein_annotations)

        protein._add(peptide_id, cast_to_int_nan_none(peptide_start, False), cast_to_int_nan_none(peptide_end, False),
                     na_to_none(peptide_annotations), cast_to_int_nan_none(single_site_or_unclear_start, False),
                     cast_to_int_nan_none(unclear_site_end, False), na_to_none(site_annotations))


def mean_counts_score(sites: List[Site], seen_count: int) -> float:
    """
    Simple scoring function that takes the mean of Site counts at a single position.
    Single sites are counted as 1, and unclear site ranges are counted as the reciprocal of possible site positions.
    Peptides where the position is seen un-glycosylated are counted as 0 in the mean.
    :param sites: list of Sites (can be single sites or unclear ranges) found at the amino acid position.
    :param seen_count: The total number of times the position has been seen in the data, including as Sites or in Peptides.
    :return: A number from 0 to 1 where 1 means 'always seen glycosylated'
    """
    if len(sites) == 0:
        return 0.0
    site_scores = [1 / site.count for site in sites]
    return math.fsum(site_scores) / seen_count


def max_score(sites: List[Site], seen_count: int) -> float:
    """
    Simple scoring function that takes the max of Site scores at a single position.
    Single sites are counted as 1, and unclear site ranges are counted as the reciprocal of possible site positions.
    Peptides where the position is seen un-glycosylated are scored as 0
    :param sites: list of Sites (can be single sites or unclear ranges) found at the amino acid position.
    :param seen_count: The total number of times the position has been seen in the data, including as Sites or in Peptides.
    :return: A number from 0 to 1 where 1 means 'seen glycosylated in nonambiguous site'
    """
    if len(sites) == 0:
        return 0.0
    site_scores = [1 / site.count for site in sites]
    return max(site_scores)
