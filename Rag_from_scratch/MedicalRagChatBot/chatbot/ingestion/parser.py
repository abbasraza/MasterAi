import logging
import re
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)

from chatbot.ingestion.cache import CacheManager


class PDFParser:

    HEADER_KEYWORDS = {
        "parameter name", "parameter", "test", "result",
        "reference value", "reference", "unit", "units",
        "chemical examination", "physical examination",
        "microscopic examination", "colour", "turbidity", "deposit",
    }

    DATE_VALUE_FORMATTED = (
        r"(\d{{1,2}}[-/][A-Za-z]{{3,9}}[-/]\d{{4}}"
        r"|\d{{1,2}}/\d{{1,2}}/\d{{4}}"
        r"|\d{{4}}-\d{{2}}-\d{{2}})"
    )

    DATE_VALUE_RAW = (
        r"(\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{4}"
        r"|\d{1,2}/\d{1,2}/\d{4}"
        r"|\d{4}-\d{2}-\d{2})"
    )

    def __init__(self, cache_dir: Path = Path("./json_cache")):
        self.cache   = CacheManager(cache_dir)
        self.converter = self._build_converter()

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def parse(self, pdf_path: Path) -> dict:
        if self.cache.exists(pdf_path):
            print(f"Cache hit -> {pdf_path.name}")
            return self.cache.load(pdf_path)

        print(f"Parsing  -> {pdf_path.name}")
        result = self._extract(pdf_path)
        self.cache.save(pdf_path, result)
        print(f"Cached   -> {pdf_path.name}")
        return result

    # -------------------------------------------------------------------------
    # Docling setup
    # -------------------------------------------------------------------------

    def _build_converter(self) -> DocumentConverter:
        try:
            options = PdfPipelineOptions(
                do_table_structure=True,
                do_ocr=True,
                generate_page_images=True,
                generate_table_images=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE,
                    do_cell_matching=True,
                ),
            )
        except Exception:
            options = PdfPipelineOptions(
                do_table_structure=True,
                do_ocr=True,
            )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )

    # -------------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------------

    def _extract(self, pdf_path: Path) -> dict:
        doc      = self.converter.convert(str(pdf_path)).document
        markdown = doc.export_to_markdown()
        print("Extracted markdown:", markdown)  # debug print
        dates    = self._extract_all_dates(markdown)

        # -------------------------------------------------------------------------
        # Document overview
        # -------------------------------------------------------------------------
        print(f"\n{'=' * 80}")
        print(f"PARSING: {pdf_path.name}")
        print(f"{'=' * 80}")
        print(f"  Pages  : {len(doc.pages)}")
        print(f"  Tables : {len(doc.tables)}")

        # -------------------------------------------------------------------------
        # Dates
        # -------------------------------------------------------------------------
        print(f"\n  Dates found:")
        for key, val in dates.items():
            print(f"    {key:20} : {val or '(not found)'}")

        output = {
            "report_date":     dates["report_date"],
            "collection_date": dates["collection_date"],
            "source_file":     pdf_path.name,
            "sections":        {},
        }

        # -------------------------------------------------------------------------
        # Headings
        # -------------------------------------------------------------------------
        headings = self._extract_headings(markdown)
        print(f"\n  Headings found ({len(headings)}):")
        if headings:
            for i, h in enumerate(headings):
                print(f"    [{i}] {h}")
        else:
            print("    (none)")

        # # ── Priority 1: Chart-based report (ESR etc) ───────────────────────────
        # if self._is_chart_report(markdown):
        #     logger = logging.getLogger(__name__)
        #     logger.info(f"Detected chart report: {pdf_path.name}")
        #     data = self._parse_chart_report(markdown, pdf_path.name)
        #     output["sections"].update(data["sections"])
        #     return output

        # ── Check for free-text report FIRST ──────────────────────────────────
        if not doc.tables or self._is_free_text_report(markdown):
            logger = logging.getLogger(__name__)

            if self._is_free_text_report(markdown):
                logger.info(f"Detected free-text report: {pdf_path.name}")
                free_text_data = self._parse_free_text_report(
                    markdown    = markdown,
                    source_file = pdf_path.name,
                )
                output["sections"].update(free_text_data["sections"])
            else:
                logger.warning(f"No tables and not a known free-text format: {pdf_path.name}")

            return output

        # -------------------------------------------------------------------------
        # Tables
        # -------------------------------------------------------------------------
        print(f"\n  Tables ({len(doc.tables)}):")

        for idx, table in enumerate(doc.tables):
            print(f"\n  --- Table {idx} ---")

            try:
                df = table.export_to_dataframe(doc)
            except Exception as e:
                print(f"    Export failed: {e}")
                continue

            if df is None or df.empty:
                print(f"    (empty)")
                continue

            # Raw dataframe info
            print(f"    Shape   : {df.shape}")
            print(f"    Columns : {list(df.columns)}")
            print(f"    Raw rows:")
            for row_idx, row in df.iterrows():
                print(f"      {list(row)}")

            # What our parser makes of it
            hint                     = headings[idx] if idx < len(headings) else ""
            name, tests, col_date    = self._table_to_tests(df, hint)

            print(f"\n    Section hint   : '{hint}'")
            print(f"    Parsed section : '{name}'")
            print(f"    Parsed date    : '{col_date or '(not found)'}'")
            print(f"    Parsed tests ({len(tests)}):")

            if tests:
                print(f"      {'Test':<35} {'Value':<10} {'Unit':<8} {'Range':<15} {'Flag':<6} Abnormal")
                print(f"      {'-'*35} {'-'*10} {'-'*8} {'-'*15} {'-'*6} --------")
                for t in tests:
                    print(
                        f"      {t['test_name']:<35} "
                        f"{t['value']:<10} "
                        f"{t['unit']:<8} "
                        f"{t['reference_range']:<15} "
                        f"{t['flag'] or 'none':<6} "
                        f"{t['is_abnormal']}"
                    )
            else:
                print("      (no tests parsed)")

            # Update output
            if col_date and not output["report_date"]:
                output["report_date"] = col_date

            if tests:
                if name in output["sections"]:
                    name = f"{name} ({idx})"
                output["sections"][name] = tests

        # -------------------------------------------------------------------------
        # Final output summary
        # -------------------------------------------------------------------------
        print(f"\n  Output summary:")
        print(f"    report_date     : {output['report_date']     or '(not found)'}")
        print(f"    collection_date : {output['collection_date'] or '(not found)'}")
        print(f"    sections        : {list(output['sections'].keys())}")
        print(f"    total tests     : {sum(len(t) for t in output['sections'].values())}")
        print(f"{'=' * 80}\n")

        return output

    # -------------------------------------------------------------------------
    # Date extraction
    # -------------------------------------------------------------------------

    def _extract_all_dates(self, markdown: str) -> dict:
        dates = {"report_date": "", "collection_date": "", "ordered_date": ""}

        LABEL_THEN_DATE = r"{label}[\s.:_-]*\n?\s*" + self.DATE_VALUE_FORMATTED

        patterns = {
            "report_date": [
                LABEL_THEN_DATE.format(label=r"Reporting\s*Date(?:Time)?"),
                LABEL_THEN_DATE.format(label=r"Verified\s*On"),
                LABEL_THEN_DATE.format(label=r"Report\s*Date"),
                LABEL_THEN_DATE.format(label=r"Reported"),
            ],
            "collection_date": [
                LABEL_THEN_DATE.format(label=r"Collection\s*Date(?:Time)?"),
                LABEL_THEN_DATE.format(label=r"Received\s*in\s*Lab"),
                LABEL_THEN_DATE.format(label=r"Collected"),
                LABEL_THEN_DATE.format(label=r"Sample\s*(?:Collected|Received)"),
            ],
            "ordered_date": [
                LABEL_THEN_DATE.format(label=r"Ordered\s*On"),
                LABEL_THEN_DATE.format(label=r"Order\s*Date"),
                LABEL_THEN_DATE.format(label=r"Registration\s*Date"),
            ],
        }

        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, markdown, re.IGNORECASE)
                if match:
                    dates[field] = match.group(1).strip()
                    break

        if not dates["report_date"]:
            dates["report_date"] = (
                dates["ordered_date"] or dates["collection_date"]
            )

        if not dates["report_date"]:
            match = re.search(self.DATE_VALUE_RAW, markdown)
            if match:
                dates["report_date"] = match.group(1).strip()

        return dates

    def _extract_headings(self, markdown: str) -> list[str]:
        headings = re.findall(r"^#{1,3}\s+(.+)$", markdown, re.MULTILINE)
        return [
            h.strip() for h in headings
            if not any(s in h.lower() for s in ["image", "page", "www", "dr.", "dr "])
        ]

    # -------------------------------------------------------------------------
    # Table parsing
    # -------------------------------------------------------------------------

    def _table_to_tests(self, df, hint: str = "") -> tuple[str, list[dict], str]:
        cols         = list(df.columns)
        section_name = self._parse_section_name(cols[0]) or hint or "General Lab Tests"
        roles        = self._detect_col_roles(cols)
        report_date  = roles["date"]
        tests        = []

        print(f"Parsing table for section '{section_name}' with detected roles: {roles} and report date: {report_date}")
        for _, row in df.iterrows():
            values    = list(row)
            first_val = self._parse_value(values[0]).lower()

            if first_val in self.HEADER_KEYWORDS or first_val.endswith("examination"):
                continue

            for group_idx, t_col in enumerate(roles["test"]):
                v_col = roles["value"][group_idx] if group_idx < len(roles["value"]) else None
                u_col = roles["unit"][group_idx]  if group_idx < len(roles["unit"])  else None
                r_cols = self._get_range_cols(roles, cols, group_idx)

                test_name = self._parse_value(values[t_col]) if t_col < len(values) else ""
                value     = self._parse_value(values[v_col]) if v_col is not None and v_col < len(values) else ""
                unit      = self._parse_value(values[u_col]) if u_col is not None and u_col < len(values) else ""
                ref_range = self._merge_range(values, r_cols)

                if not test_name or test_name.lower() in self.HEADER_KEYWORDS:
                    continue

                flag, is_abnormal = self._compute_flag(value, ref_range)

                tests.append({
                    "test_name":       test_name,
                    "value":           value,
                    "unit":            unit,
                    "reference_range": ref_range,
                    "flag":            flag,
                    "is_abnormal":     is_abnormal,
                })

        return section_name, tests, report_date

    def _get_range_cols(self, roles: dict, cols: list, group_idx: int) -> list[int]:
        if not roles["range"]:
            return []
        if all(isinstance(c, str) for c in cols):
            return roles["range"]
        return [roles["range"][group_idx]] if group_idx < len(roles["range"]) else []

    def _merge_range(self, values: list, r_cols: list[int]) -> str:
        parts = [self._parse_value(values[ri]) for ri in r_cols if ri < len(values)]
        parts = [p for p in parts if p]
        ref   = " ".join(parts)
        ref   = re.sub(r"\s*-\s*-\s*", " - ", ref)
        return re.sub(r"\s+", " ", ref).strip()

    # -------------------------------------------------------------------------
    # Column role detection
    # -------------------------------------------------------------------------

    def _detect_col_roles(self, cols: list) -> dict:
        roles = {"test": [], "value": [], "unit": [], "range": [], "date": ""}

        if all(isinstance(c, str) for c in cols):
            for i, col in enumerate(cols):
                clean = self._parse_column_name(col)

                if clean in {"test", "parameter", "parameter name",
                            "investigation", "analyte"}:
                    roles["test"].append(i)

                # ── Range detection (must come BEFORE value) ───────────────────
                # "Reference Value", "Reference Range", "Normal Range",
                # "Ref. Value", "Ref Range", "Normal Value"
                elif any(k in clean for k in [
                    "reference", "ref", "normal range",
                    "normal value", "interval",
                ]):
                    roles["range"].append(i)

                # ── Value/Result detection ─────────────────────────────────────
                # Plain "result", "value", "finding"
                # OR column name that looks like a date/specimen ID
                # e.g. "56415-04-06 04 Jun 2025 14:13"
                elif (
                    clean in {"result", "value", "finding"}
                    or self._looks_like_result_col(col)
                ):
                    roles["value"].append(i)
                    if not roles["date"]:
                        roles["date"] = self._extract_date_from_col(col)

                elif clean in {"unit", "units"}:
                    roles["unit"].append(i)

            return roles

        # Integer columns
        n = len(cols)
        if n >= 4:
            roles["test"].append(0)
            roles["value"].append(1)
            roles["range"].append(2)
            if n > 3:
                roles["unit"].append(3)
            if n >= 8:
                roles["test"].append(4)
                roles["value"].append(5)
                roles["range"].append(6)
                roles["unit"].append(7)
            elif n >= 6:
                roles["test"].append(4)
                roles["value"].append(5)

        return roles


    def _looks_like_result_col(self, col: str) -> bool:
        """
        Detect result columns that look like specimen IDs or dates.

        Examples:
        '56415-04-06 04 Jun 2025 14:13'   <- specimen ID + date
        'Result _4706976_ 02-May-2025'     <- result + metadata
        '26-010242300'                     <- specimen number

        Strategy:
        - Contains a date pattern  AND
        - Does NOT contain range/reference/normal/unit keywords
        """
        if not isinstance(col, str):
            return False

        col_lower = col.lower()

        # Exclude if it looks like a range/reference column
        exclude_keywords = {
            "reference", "ref", "normal", "range",
            "interval", "unit", "test", "parameter",
        }
        if any(k in col_lower for k in exclude_keywords):
            return False

        # Include if it contains a date or specimen ID pattern
        import re
        date_patterns = [
            r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}",   # 04 Jun 2025
            r"\d{1,2}[-/][A-Za-z]{3}[-/]\d{4}",  # 04-Jun-2025
            r"\d{1,2}/\d{1,2}/\d{4}",            # 04/06/2025
            r"\d{5,}-\d{2}-\d{2}",               # 56415-04-06 (specimen ID)
        ]
        for pattern in date_patterns:
            if re.search(pattern, col, re.IGNORECASE):
                return True

        return False
    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------

    def _parse_section_name(self, col) -> str:
        if not isinstance(col, str):
            return ""
        return col.split(".")[0].strip() if "." in col else ""

    def _parse_column_name(self, col) -> str:
        if not isinstance(col, str):
            return ""

        original = col

        # Only split on dot if it looks like "Section.ColumnName"
        # NOT if dot is part of a date like "14:13"
        if "." in col:
            parts      = col.split(".")
            meaningful = [p.strip() for p in parts if p.strip()]
            # Only use dot-split if result looks like a clean column name
            # not a date/specimen ID
            if meaningful and not any(
                c.isdigit() for c in meaningful[-1][:3]
            ):
                col = meaningful[-1]

        col = col.strip().lower()

        # Remove metadata noise BUT preserve date columns
        # Don't strip if it looks like a date/specimen column
        import re
        if not re.search(
            r"\d{1,2}\s+[A-Za-z]{3}|\d{5,}-\d{2}", col
        ):
            col = re.split(r"\s*_\d+", col)[0].strip()

        print(f"  Analyzing column '{original}' -> cleaned: '{col}'")
        return col

    def _parse_value(self, val) -> str:
        if val is None:
            return ""
        s = str(val).strip()
        return "" if s.lower() in {"none", "nan", ""} else s

    def _extract_date_from_col(self, col) -> str:
        if not isinstance(col, str):
            return ""

        import re
        patterns = [
            r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})",   # 04 Jun 2025
            r"(\d{1,2}-[A-Za-z]{3,9}-\d{4})",         # 04-Jun-2025
            r"(\d{1,2}/\d{1,2}/\d{4})",               # 04/06/2025
            r"(\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{4})", # existing
        ]
        for pattern in patterns:
            match = re.search(pattern, col, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _compute_flag(self, value: str, reference_range: str) -> tuple[str, bool]:
        match = re.search(r"([\d.]+)\s*-\s*([\d.]+)", str(reference_range))
        if not match or not value:
            return "", False
        try:
            val  = float(str(value).replace(",", ""))
            low  = float(match.group(1))
            high = float(match.group(2))
            if val < low:
                return "low", True
            elif val > high:
                return "high", True
            return "", False
        except (ValueError, TypeError):
            return "", False
    
    # Add these to PDFParser class

    # Keywords that indicate a free-text report (no table)
    FREE_TEXT_REPORT_KEYWORDS = [
        "no growth",
        "growth obtained",
        "final report",
        "culture method",
        "maldi-tof",
        "antibiotic susceptibility",
        "sensitivity report",
        "gram positive",
        "gram negative",
        "organism isolated",
        "no organism",
        "sterile",
        "colony",
        "fetoprotein"
    ]


    def _is_free_text_report(self, markdown: str) -> bool:
        """
        Detect if this is a free-text report (culture, sensitivity)
        with no structured table.
        """
        text_lower = markdown.lower()
        print("text_lower for free-text detection:", text_lower)  # debug print
        matches    = sum(
            1 for kw in self.FREE_TEXT_REPORT_KEYWORDS
            if kw in text_lower
        )
        print(f"Free-text report keyword matches: {matches}")  # debug print
        return matches >= 1


    def _parse_free_text_report(
        self,
        markdown:    str,
        source_file: str,
    ) -> dict:
        """
        Parse free-text reports like culture results.
        Extracts key findings as structured test entries.
        """
        import re
        logger = logging.getLogger(__name__)
        logger.info(f"Parsing as free-text report: {source_file}")

        text       = markdown.strip()
        text_lower = text.lower()

        # ── Determine result ───────────────────────────────────────────────────
        if "no growth" in text_lower:
            result      = "No Growth"
            is_abnormal = False
            flag        = ""
        elif "growth" in text_lower:
            # Extract organism if mentioned
            organism_match = re.search(
                r"(growth\s+of\s+|organism[:\s]+|isolated[:\s]+)"
                r"([A-Za-z\s]+?)(?:\.|,|\n|$)",
                text, re.IGNORECASE
            )
            result = (
                organism_match.group(2).strip()
                if organism_match
                else "Growth Detected"
            )
            is_abnormal = True
            flag        = "abnormal"
        else:
            result      = "See report"
            is_abnormal = False
            flag        = ""

        # ── Extract specimen type ──────────────────────────────────────────────
        specimen_match = re.search(
            r"specimen[:\s]+([A-Za-z\s\(\)]+?)(?:\n|$|collection)",
            text, re.IGNORECASE
        )
        specimen = (
            specimen_match.group(1).strip()
            if specimen_match
            else "Unknown"
        )

        # ── Extract blood type if present ─────────────────────────────────────
        blood_type_match = re.search(
            r"(blood\s*\([A-Za-z\s]+\)|urine|csf|wound|sputum|stool)",
            text, re.IGNORECASE
        )
        blood_type = (
            blood_type_match.group(1).strip()
            if blood_type_match
            else ""
        )

        test_name = f"Culture"
        if blood_type:
            test_name = f"Culture - {blood_type}"
        elif specimen and specimen.lower() not in {"culture", "unknown"}:
            test_name = f"Culture - {specimen}"

        # ── Extract reference value ────────────────────────────────────────────
        ref_match = re.search(
            r"reference\s+value[:\s]+([^\n]+)",
            text, re.IGNORECASE
        )
        reference = (
            ref_match.group(1).strip()
            if ref_match
            else "No Growth/Normal flora isolated"
        )

        # ── Extract comments ───────────────────────────────────────────────────
        comment_match = re.search(
            r"(?:final\s+report|result)[:\s]+([^\n]+(?:\n(?!reference|comment|culture)[^\n]+)*)",
            text, re.IGNORECASE
        )
        comment = (
            comment_match.group(1).strip()
            if comment_match
            else result
        )

        tests = [{
            "test_name":       test_name,
            "value":           result,
            "unit":            "",
            "reference_range": reference,
            "flag":            flag,
            "is_abnormal":     is_abnormal,
            "comment":         comment[:200],   # truncate long comments
        }]

        return {
            "sections": {
                "Culture & Sensitivity": tests
            }
        }
        # Add to PDFParser class

        # Keywords that indicate a chart-based single-value report
    CHART_REPORT_KEYWORDS = [
        "esr",
        "erythrocyte sedimentation",
        "mm/1st",
        "westergren",
    ]

    # Known single-value test patterns with their reference parsing
    SINGLE_VALUE_TESTS = {
        "esr": {
            "full_name": "ESR (Erythrocyte Sedimentation Rate)",
            "unit":      "mm/1st Hr",
            "section":   "Department of Hematology",
        },
        "hba1c": {
            "full_name": "HbA1c",
            "unit":      "%",
            "section":   "Department of Chemical Pathology",
        },
        "crp": {
            "full_name": "C-Reactive Protein (CRP)",
            "unit":      "mg/L",
            "section":   "Department of Immunology",
        },
    }


    def _is_chart_report(self, markdown: str) -> bool:
        """Detect chart-based single value reports like ESR."""
        text_lower = markdown.lower()
        return any(kw in text_lower for kw in self.CHART_REPORT_KEYWORDS)


    def _parse_chart_report(self, markdown: str, source_file: str) -> dict:
        """
        Parse chart-based single-value reports.

        Handles:
        ESR: 41 mm/1st Hr  Reference: Normal (=15)
        HbA1c: 6.5%        Reference: Normal (<5.7)
        """
        import re
        logger = logging.getLogger(__name__)
        logger.info(f"Parsing as chart report: {source_file}")

        text       = markdown.strip()
        text_lower = text.lower()

        # ── Detect which test this is ──────────────────────────────────────────
        test_key  = None
        test_info = None
        for key, info in self.SINGLE_VALUE_TESTS.items():
            if key in text_lower:
                test_key  = key
                test_info = info
                break

        if not test_info:
            logger.warning(f"Unknown chart report type: {source_file}")
            test_info = {
                "full_name": "Unknown Test",
                "unit":      "",
                "section":   "General Lab Tests",
            }

        # ── Extract reference range ────────────────────────────────────────────
        reference_range = ""
        ref_patterns = [
            # "Normal (=15)" or "Normal (<=15)"
            r"normal\s*\(<=?\s*([\d.]+)\)",
            # "Normal (<15)"
            r"normal\s*\(<\s*([\d.]+)\)",
            # "Normal: up to 15"
            r"normal[:\s]+up\s+to\s+([\d.]+)",
            # "Reference: <= 15" or "Reference: < 15"
            r"reference[:\s]+<=?\s*([\d.]+)",
            # "0 - 15" standard range
            r"([\d.]+)\s*[-–]\s*([\d.]+)",
        ]

        upper_bound = None
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.lastindex == 2:
                    # Range pattern "0 - 15"
                    reference_range = f"{match.group(1)} - {match.group(2)}"
                    upper_bound     = float(match.group(2))
                else:
                    # Upper bound only "= 15" or "< 15"
                    upper_bound     = float(match.group(1))
                    reference_range = f"0 - {upper_bound}"
                break

        # ── Extract the actual result value ───────────────────────────────────
        # Strategy: find all numbers, pick the one that appears
        # immediately after the test name heading in the text
        all_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
        result      = None

        # Filter out obvious non-result numbers:
        # - Phone numbers (long sequences)
        # - Years (4 digits starting with 19/20)
        # - Specimen IDs (appear in parentheses like (56436-26-05))
        # - Reference bound itself
        def is_likely_result(num_str: str) -> bool:
            n = float(num_str)
            # Skip if looks like year
            if 1900 <= n <= 2100:
                return False
            # Skip if looks like phone number component (too large)
            if n > 9999:
                return False
            # Skip if it IS the reference bound
            if upper_bound and n == upper_bound:
                return False
            return True

        # Find result: number appearing right after test name
        # e.g. "ESR\n41\n46\n42" -> 41 is the result (first after heading)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            if test_key and test_key.lower() in line.lower():
                # Look at next few lines for a numeric result
                for j in range(i + 1, min(i + 5, len(lines))):
                    num_match = re.match(r"^(\d+(?:\.\d+)?)\s*$", lines[j])
                    if num_match and is_likely_result(num_match.group(1)):
                        result = num_match.group(1)
                        break
                break

        # Fallback: first valid number in all_numbers
        if not result:
            valid_numbers = [n for n in all_numbers if is_likely_result(n)]
            if valid_numbers:
                result = valid_numbers[0]

        if not result:
            logger.warning(f"Could not extract result value from: {source_file}")
            result = "Not extracted"

        # ── Compute flag ───────────────────────────────────────────────────────
        flag, is_abnormal = self._compute_flag(result, reference_range)

        # ── Extract unit from text ─────────────────────────────────────────────
        unit = test_info["unit"]
        unit_match = re.search(
            r"(mm/1st\s*h(?:r|our)?|mm/hr|%|g/dl|mg/l)",
            text, re.IGNORECASE
        )
        if unit_match:
            unit = unit_match.group(1).strip()

        tests = [{
            "test_name":       test_info["full_name"],
            "value":           result,
            "unit":            unit,
            "reference_range": reference_range,
            "flag":            flag,
            "is_abnormal":     is_abnormal,
            "comment":         f"Result extracted from chart-based PDF report.",
        }]

        logger.info(
            f"Chart report parsed: "
            f"test={test_info['full_name']} | "
            f"value={result} | "
            f"range={reference_range} | "
            f"flag={flag}"
        )

        return {
            "sections": {
                test_info["section"]: tests
            }
        }