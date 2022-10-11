## Introduction
The project gives a solution for an AI-based cloud service, which aims to automatize the production of character animation.

Generally, the project is consisted of three parts :
1. Unity client
   - an interaction interface for users
   - generate and parse a BVH file
   - visualize the motion data in a BVH file 
2. Triton Server
   - providing inference service using deployed neural network models
3. Request Handler
   - communicate with Unity client and Triton
   - data processing
## Use case
Two functionalities are involved : 
- Generate a BVH file<br/>For an input video, a BVH file will be generated which could be directly used in animation editors(e.g., Blender).
- Visualize a BVH file<br/>For a loaded BVH file, motion data will be extracted and mapped to an avatar, then the animation can play
## Deployment
In the folder 'backend', two types of backend are provided. You can deploy the backend on a local machine or a remote server, depending on your need. 
## Deploy locally
For a quick use of this project, Triton is not needed, follow : 
1. Install dependencies
<br/>Request Handler runs in a python environment (conda is recommended), make sure following packages are installed properly:
     - numpy
     - pytorch
     - opencv-python, opencv-python-contrib
     - flask
2. Run the backend
<br/>simply start the flask server by :
     ```
     python main_local.py
     ```
     Now the bakcend is ready, you can interact with the system through the Unity client provided.
## Deploy with Triton
If you want to run the project as a cloud service using Triton Inference Server, follow :
1. Start a TensorRT environment
<br/>Please follow https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/running.html#running to setup TensorRT. 
<br/>A docker image is highly recommended.
2. ONNX to TensorRT
<br/>For faster inference, you need to convert the models in 'onnxModels' into .engine files in a TensorRT environment.
<br/>For YOLO, run :
     ```
     trtexec --onnx=yolov3_spp_-1_608_608_dynamic_folded.onnx 
     --explicitBatch 
     --saveEngine=yolov3_spp_-1_608_608_dynamic_folded.engine 
     --workspace=10240 --fp16 --verbose 
     ```
     For FastPose, run :
     ```
     trtexec --onnx=FastPose_-1_3_256_192_dynamic.onnx 
     --saveEngine=FastPose_-1_3_256_192_dynamic.engine 
     --workspace=10240 --verbose 
     --minShapes=input:1x3x256x192 
     --optShapes=input:1x3x256x192 
     --maxShapes=input:128x3x256x192 
     --shapes=input:1x3x256x192 
     --explicitBatch
     ```
     Now you get two .engine files, which will be used laterly. 
     <br/>Note that the VideoPose3D model is not converted, as there is some problem in converting that I haven't figured it out.

