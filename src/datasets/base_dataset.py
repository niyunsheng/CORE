from torch.utils.data import Dataset
from PIL import Image 

class BaseDataset(Dataset):
    def __init__(self, root, transform=None, num_classes=2):
        super(BaseDataset,self).__init__()
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        assert transform is not None, "transform is None"

    def __getitem__(self,idx):
        img_path = self.imgs[idx][0]
        label = self.imgs[idx][1]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.imgs)