# DL-UIA - Deep Learning in Ultrasound Image Analysis 
## 3D Surface Mesh Estimation Dataset
This is the training dataset for the open challenge 3D Surface Mesh Estimation for CVPR'23 workshop: DL-UIA - Deep Learning in Ultrasound Image Analysis.
Workshop website: https://www.cvpr2023-dl-ultrasound.com/ 

### Terms of and Conditions of Access
[Participant_FULLNAME] (the "Participant") has requested permission to use the DL-UIA dataset (the "Dataset") for CVPR 2023 Workshop. In exchange for such permission, Participant hereby agrees to adhere to the following terms and conditions:
* Participant shall use the Dataset only for non-commercial research, publication and educational purposes.
* Participant may provide research associates, and colleagues with access to the Dataset provided that they first agree to be bound by Terms of and Conditions of Access.
* Redistribution or transferring of competition data or data links is not allowed during or after the competition. Participants should use the data only for this competition and publication in CVPR 2023. If the participant wants to access, redistribute, or transfer the Dataset for any other purposes, he or she should contact the DL-UIA committee for permission.
* The DL-UIA committee and the DarkVision Technologies Inc. reserves the right to terminate the Participant's access to the Dataset at any time.
* If the Participant is employed by a for-profit, commercial entity, the Participant’s employer shall also be bound by these Terms of and Conditions of Access, and the Participant hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.
* The law of the Province of British Columbia shall apply to all disputes under this agreement.

### Problem Statement
For a given input volumetric ultrasound image, give a corresponding estimation of the surface mesh.

Training dataset:
* 89 raw volumetric ultrasound images
* 5 reference 3D meshes, corresponding to the volumetric scans 001-005

Raw Volumetric Image Meta Info (for visualization and mesh alignment):
origin - (0, 0, 0)
spacing: (0.49479, 0.49479, 0.3125)
data type: unsigned short int
volume dimension: (768, 768, 1280)

Data Visualization:
We recommend using ParaView 5.9.1 for visualization purposes.
