# Internship_sarcopenia


**Neck Navigator:**

A U-Net model to select the C3 vetebra from the 3D CT scan (NIFTI file)

<img src="https://user-images.githubusercontent.com/60819221/135115131-b39765e8-a919-4b53-82a1-166d3b13dadb.png" width="600" height="300">

**Muscle Mapper:**

An FCN neural network to automatically delineate the paravertebral and sternocleidomastoid muscles at the level of the C3 vertebra. 

![pred_vs_gt](https://user-images.githubusercontent.com/60819221/135115254-eaa2a0ab-b22a-465f-a792-edb948959e1d.png)


**Inference:**

A script to apply Neck Navigator and Muscle Mapper to all the patients in a directory and write the extracted muscel characteristics to an excel file.
