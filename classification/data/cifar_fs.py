from torchmeta.datasets import CIFARFS
from torchmeta.transforms import Categorical, ClassSplitter
from torchvision.transforms import ToTensor
from torchmeta.utils.data import BatchMetaDataLoader

from utils.paths import datasets_path
from utils.misc import set_seed

def get_loader(batch_size, ways, shots, test_shots,
        meta_split='train', download=True, num_workers=4):

    dataset = CIFARFS(datasets_path,
            num_classes_per_task=ways,
            transform=ToTensor(),
            target_transform=Categorical(num_classes=ways),
            meta_split=meta_split,
            download=download)

    dataset = ClassSplitter(dataset,
            shuffle=(meta_split != 'test'),
            num_train_per_class=shots,
            num_test_per_class=test_shots)

    return BatchMetaDataLoader(dataset,
            batch_size=batch_size, num_workers=num_workers)
