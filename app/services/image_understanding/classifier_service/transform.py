from PIL import Image
from torchvision import transforms


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_size = max(width, height)
        pad_w = max_size - width
        pad_h = max_size - height
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        fill_color = (self.fill, self.fill, self.fill) if isinstance(self.fill, int) else self.fill
        output = Image.new(image.mode, (max_size, max_size), fill_color)
        output.paste(image, (pad_left, pad_top))
        return output


class NormalTransform:
    def __init__(self, input_size: int = 512):
        self.val_transform = transforms.Compose(
            [
                PadToSquare(fill=0),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, image: Image.Image):
        return self.val_transform(image)
