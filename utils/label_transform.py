def labels_to_indices(labels):
    return {label : idx for idx, label in enumerate(labels)}

def indices_to_labels(labels):
    return {idx : label for idx, label in enumerate(labels)}

class LabelTransform:
    def __init__(self, labels):
        self.labels = labels
        self.label_to_index = labels_to_indices(self.labels)
        self.index_to_label = indices_to_labels(self.labels)

    def label_transform(self, data):
        func = self.label_to_index
        return list(map(lambda label : func[label], data))
    
    def index_transform(self, data):
        func = self.index_to_label
        return list(map(lambda index : func[index], data))
        