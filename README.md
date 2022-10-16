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
  Interpolation and smoothing are supported.
  <p align="center">
  <img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif" width="300" right="200"/>
  <img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif" width="300" /> 
</p>
- Visualize a BVH file<br/>For a loaded BVH file, motion data will be extracted and mapped to an avatar, then the animation can play
## Deployment
In the folder 'backend', two types of backend are provided. You can deploy the backend either on a local machine or a remote server, depending on your need. 
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
     Now you get two .engine files, which will be used later. 
     <br/>Note: 
     - VideoPose3D model is not converted, as there is some problem in converting that I haven't figured it out.
     - .engine files are not provided here since the conversion from ONNX to .engine is dependent on what model of GPU you are using.
     
3. Create model repository
<br/>To run on Triton, the three neural network models should be organized in a specific way :

     The folder ```backend/triton/Models``` is ready for being a repository, you just need to rename the obtained engine files as 'model.plan' and put them into ```backend/triton/Models/YOLO```, ```backend/triton/FastPose``` and ```backend/triton/Models/VideoPose3D```, respectively.
     <br/>
     <br/>```config.pbtxt``` specifies the running configuration of a model. They have been written properly, so you don't need to modify them.
4. Start Triton Inference Server
<br/>Check https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/ and get a released docker image of Triton, make sure that it matches your OS and CUDA. Then, start Triton as a docker container :
     ```
     sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/rlt/Desktop/trition:/models nvcr.io/nvidia/tritonserver:<xx.xx>-py3 tritonserver --model-repository=/Models
     ```
     ```/Models``` is the path of the model repository you created in Step3.   
<br/>For detailed description of how to deploy an AI application based on Triton, please refer to the official documentation https://github.com/triton-inference-server/server.
