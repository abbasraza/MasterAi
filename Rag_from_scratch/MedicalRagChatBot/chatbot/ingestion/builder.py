from langchain_core.documents import Document


class DocumentBuilder:

    def build(self, report: dict) -> list[Document]:
        documents       = []
        report_date     = report.get("report_date",     "Unknown")
        collection_date = report.get("collection_date", "Unknown")
        source_file     = report.get("source_file",     "unknown")
        all_tests       = []
        abnormal_tests  = []

        for section_name, tests in report.get("sections", {}).items():
            for test in tests:
                is_abnormal  = test.get("is_abnormal", False)
                flag         = test.get("flag",        "")
                comment      = test.get("comment",     "")
                abnormal_tag = f" [ABNORMAL - {flag.upper()}]" if is_abnormal else ""

                # ── Build page content ─────────────────────────────────────
                content_parts = [
                    f"Test: {test['test_name']}{abnormal_tag}",
                    f"Result: {test.get('value', '')} {test.get('unit', '')}".strip(),
                    f"Reference Range: {test.get('reference_range') or 'N/A'}",
                ]

                if comment:
                    content_parts.append(f"Comment: {comment}")

                content_parts += [
                    f"Section: {section_name}",
                    f"Report Date: {report_date}",
                    f"Collection Date: {collection_date}",
                    f"Source: {source_file}",
                ]

                documents.append(Document(
                    page_content = "\n".join(content_parts),
                    metadata     = {
                        "doc_type":        "individual_test",
                        "test_name":       test.get("test_name", "").lower(),
                        "value":           test.get("value",           ""),
                        "unit":            test.get("unit",            ""),
                        "reference_range": test.get("reference_range", ""),
                        "is_abnormal":     is_abnormal,
                        "flag":            flag,
                        "comment":         comment,
                        "section":         section_name,
                        "report_date":     report_date,
                        "collection_date": collection_date,
                        "source_file":     source_file,
                    }
                ))

                all_tests.append(test)
                if is_abnormal:
                    abnormal_tests.append((section_name, test))

        documents.append(self._build_summary(
            report_date     = report_date,
            collection_date = collection_date,
            source_file     = source_file,
            all_tests       = all_tests,
            abnormal_tests  = abnormal_tests,
        ))

        return documents

    # -------------------------------------------------------------------------

    def _build_summary(
        self,
        report_date:     str,
        collection_date: str,
        source_file:     str,
        all_tests:       list,
        abnormal_tests:  list,
    ) -> Document:

        # ── All results text ───────────────────────────────────────────────
        all_text = "\n".join(
            "  {}: {} {}{}".format(
                t.get("test_name", ""),
                t.get("value",     ""),
                t.get("unit",      ""),
                f"  ABNORMAL {t['flag'].upper()}" if t.get("is_abnormal") else "",
            )
            for t in all_tests
        ) or "  None"

        # ── Abnormal results text ──────────────────────────────────────────
        abnormal_text = "\n".join(
            "  [{}] {}: {} {} (Range: {}) {}{}".format(
                section,
                t.get("test_name",       ""),
                t.get("value",           ""),
                t.get("unit",            ""),
                t.get("reference_range", "?"),
                t.get("flag",            "").upper(),
                f" | {t['comment']}" if t.get("comment") else "",
            )
            for section, t in abnormal_tests
        ) or "  None"

        # ── Comments from all tests ────────────────────────────────────────
        all_comments = [
            f"  [{t.get('test_name', '')}]: {t['comment']}"
            for t in all_tests
            if t.get("comment")
        ]
        comments_text = "\n".join(all_comments)

        # ── Build page content ─────────────────────────────────────────────
        content_parts = [
            "Lab Report Summary",
            f"Report Date: {report_date}",
            f"Collection Date: {collection_date}",
            f"Source: {source_file}",
            f"Total Tests: {len(all_tests)}",
            f"Abnormal Count: {len(abnormal_tests)}",
            "",
            "All Results:",
            all_text,
            "",
            "Abnormal Results:",
            abnormal_text,
        ]

        if comments_text:
            content_parts += ["", "Comments:", comments_text]

        return Document(
            page_content = "\n".join(content_parts),
            metadata     = {
                "doc_type":        "report_summary",
                "report_date":     report_date,
                "collection_date": collection_date,
                "source_file":     source_file,
                "abnormal_count":  len(abnormal_tests),
                "total_tests":     len(all_tests),
            }
        )