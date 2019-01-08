# CS337CG-3D-ShapeNets
project for CS337 Computer Graphic

Based on paper *3D ShapeNets: A Deep Representation for Volumetric Shape Modeling*

Download the original off mesh data. Use utils/write_input_data.m too convert the mesh representation into volumes.

The file run_finetuning.m can be just ignored since it is time consuming but merely sometimes improve performance slightly.
## About the files
      1. The root folder contains interfaces for training and testing.
      2. kFunction.cu and kFunction2.cu provide a 3D cuda convolution routine based on developed by Alex Krizhevsky.
      3. The folder "generative" is for probablistic CDBN training.
      4. The folder "bp" does discriminative fine-tuning for 3D mesh classification and retrieval.
      5. The folder "3D" involves 3D computations like TSDF and Rendering.
      6. The folder "voxelization" is a toolbox to convert the mesh model to a volume representation. 
      7. We provide a classifcation of volumetric shapes using Deep Neural Networks in utils.py.
      8. net2netwider.m, net2netdeeper.m etc. are files that constitutes the new architecture.
      9. The files at the beginning of "run" are our training files for different models.
