import Augmentor
import os
import time
from tqdm import tqdm

path_to_data = "./images/"
output_path = "./images/output/"

# Create a pipeline
p = Augmentor.Pipeline(path_to_data)


# Add some operations to an existing pipeline.

# First, we add a horizontal flip operation to the pipeline:
# p.flip_left_right(probability=0.7)
p.gaussian_distortion(0.7, 10, 10, 0.7,"bell","out")
# p.histogram_equalisation(probability=1.0)

# # Now we add a vertical flip operation to the pipeline:
# p.flip_top_bottom(probability=0.8)

# # Add a rotate90 operation to the pipeline:
# p.rotate90(probability=0.1)
# Here we sample 100,000 images from the pipeline.

# It is often useful to use scientific notation for specify
# large numbers with trailing zeros.
num_of_samples = int(3000)

# Now we can sample from the pipeline:
p.sample(num_of_samples)

time.sleep(5)
##### RENAME #######
i=0
for filename in tqdm(os.listdir(output_path)):
    print(filename[16:])

    dst =filename[16:]
    src =output_path + filename
    dst =output_path+ dst
    os.rename(src, dst)
    i += 1
