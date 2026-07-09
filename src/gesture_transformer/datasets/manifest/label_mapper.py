class LabelMapper:
    def converter_label(self, label: str) -> str:
        """
        Converts a label to a standardized format.
        For example, it can convert to lowercase, replace spaces with underscores, etc.
        """
        # Example conversion: convert to lowercase and replace spaces with underscores
        return label.lower().replace(" ", "_")