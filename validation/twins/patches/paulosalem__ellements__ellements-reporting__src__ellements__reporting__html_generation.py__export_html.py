    def export_html(self, content: str, filename: str = "report.html") -> Path:
        """Write *content* as an HTML file under :attr:`output_dir`."""
        base = Path(self.output_dir).resolve()
        path = (base / filename).resolve()
        if not path.is_relative_to(base):
            raise ValueError("filename escapes output_dir")
        path.write_text(content, encoding="utf-8")
        return path
