import os
from PIL import Image, ExifTags
import pyheif

def heic_to_jpg(path):
    if path.endswith(".HEIC"):
        heif_file = pyheif.read(path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
    else:
        image = Image.open(path)
        try:
            # Grab orientation value.
            image_exif = image._getexif()
            image_orientation = image_exif[274]
            # Rotate depending on orientation.
            if image_orientation == 3:
                image = image.rotate(180)
            if image_orientation == 6:
                image = image.rotate(-90)
            if image_orientation == 8:
                image = image.rotate(90)
        except:
            pass
    return image



if __name__=='__main__':
    root_dir = "/home/ahmadob/dataset/facerecognition_dataset/"
    source_dir = os.path.join(root_dir, 'train_set')
    destination_dir = os.path.join(root_dir, 'train_jpeg_set')
    print(os.listdir(source_dir))


    for src_dir in os.listdir(source_dir):
        src_dir_path = os.path.join(source_dir, src_dir)
        dst_dir_path = os.path.join(destination_dir, src_dir)
        os.makedirs(dst_dir_path, exist_ok=True)

        print("Processing {}".format(src_dir_path))
        for ind, src_img in enumerate(os.listdir(src_dir_path)):
            src_img_path = os.path.join(src_dir_path, src_img)

            img = heic_to_jpg(src_img_path)
            dst_img_path = os.path.join(dst_dir_path, str(ind)+".jpg")
            img.save(dst_img_path)
        print("Processed and saved at {}".format(dst_dir_path))