from detector import get_fasterrcnn_model, get_transform, make_detections_on_test_set
from dataset import PeopleDataset

model = get_fasterrcnn_model()
model.eval()
sequences_to_evaluate = ['sequence-1']
dataset = PeopleDataset(None)
_, dataset_test = dataset.train_test_split(0.99)
#train_dataset.transforms = get_transform(train=True)
dataset_test.transforms = get_transform(train=False)
make_detections_on_test_set(model, dataset_test, sequences_to_evaluate, shuffle=False)