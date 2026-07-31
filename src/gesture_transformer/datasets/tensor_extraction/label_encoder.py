class LabelEncoder:
    def __init__(self, labels: list[str]):
        unique_labels = sorted(set(labels))

        self.labels = unique_labels
        self.label_to_index = {
            label: index
            for index, label in enumerate(unique_labels)
        }
        self.index_to_label = {
            index: label
            for label, index in self.label_to_index.items()
        }

    def encode(self, label: str) -> int:
        if label not in self.label_to_index:
            raise ValueError(f"Unknown label: {label}")

        return self.label_to_index[label]

    def decode(self, index: int) -> str:
        if index not in self.index_to_label:
            raise ValueError(f"Unknown label index: {index}")

        return self.index_to_label[index]

    def mapping(self) -> dict[str, int]:
        return self.label_to_index