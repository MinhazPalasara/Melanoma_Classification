__author__ = 'Minhaz Palasara'


from melanoma.data_processor.suborder_preprocessor import DataMetadata, ProcessData
from melanoma.data_processor.extract_data import ExtractImage, ExtractSuborder
from melanoma.data_processor.suborder_preprocessor import GenerateData

# ids of layers to be packaged in suborder
suborder = [1, 2, 3, 4, 5, 6, 7]

data_path = "/home/minhazpalasara/segmented_image/segmented_all_images/"
output_path = "/home/minhazpalasara/segmented_image/segmented_all_images/"

# Four different data pre-processing operations are supported

# To extract the dataset metadata
# processor = DataMetadata()
# processor.loadSize(data_path=data_path)
# processor.getMetadata()

# To append the images with black pixels
# data_generator = ProcessData(676, 804)
# data_generator.appendData(data_path=data_path, output_path=output_data_path)
# data_generator.upsampleData(data_path=data_path, output_path=output_data_path)

# To generate suborders in .h5 files
# extract = ExtractSuborder()
# extract.extract_suborder(data_path=data_path, output_path=output_path, suborder_list=suborder)

# .h5 file to .jpeg files
extract = ExtractImage()
extract.extractImage(data_path=data_path, output_path=output_path)

# Data Augmentation
# extract = GenerateData()
# extract.generateRotatedData(data_path, output_path)

