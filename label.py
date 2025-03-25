from dataset import VQAFeatureDataset

def labeling_for_label_score(train_dset: VQAFeatureDataset, base_score = 0.0):
    grouped_data = {}
    for i, data in enumerate(train_dset.entries):
        data['serial'] = i
        if data['answer']['labels'] is None:
            key = (None, None)
        else:
            labels = data['answer']['labels'].tolist()
            scores = data['answer']['scores'].tolist()

            filtered_labels = [label for label, score in zip(labels, scores) if score > base_score]
            filtered_scores = [score for label, score in zip(labels, scores) if score > base_score]

            if filtered_labels.__len__() == 0:
                filtered_labels = labels
                filtered_scores = scores

            labels = tuple(sorted(filtered_labels))
            scores = tuple(sorted(filtered_scores))
            
            key = (labels, scores)

        if key not in grouped_data:
            grouped_data[key] = []
        
        grouped_data[key].append(data)
    train_dset.grouped_data = grouped_data
