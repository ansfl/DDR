# Model-Based and Neural-Aided Approaches for Dog Dead Reckoning


# Abstract
<div align="justify">
  Modern canine applications include medical and service roles while robotic legged dogs function as autonomous platforms for high-risk industrial inspection, disaster response, and search and rescue operations. For both,   accurate positioning remains a challenge due to the cumulative drift inherent in inertial sensing. To bridge this gap, in this work we propose three algorithms using only inertial sensors for accurate positioning,     referred to as dog dead reckoning (DDR). To evaluate our approaches we  designed DogMotion, a wearable unit for canine data recording. Using DogMotion we recorded a dataset of 13 minutes. In addition, we employed a robotic legged dog dataset with a duration of 116 minutes.  Across the two distinct datasets we demonstrate that our neural-aided methods consistently outperform the model-based approach with an absolute distance error of less than 10 [%]. As such, we offer a lightweight and low-cost positioning solution for both biological dogs and legged robotic dogs. A codebase implementing the described framework and our associated dataset have been made publicly available.
</div>


# Dataset
<div align="justify">
The study evaluates the proposed Dog Dead Reckoning (DDR) framework using two distinct datasets  of synchronized motion data. The first is a custom-recorded Dog Dataset, collected using the DogMotion wearable unit—a Raspberry Pi Zero-based system equipped with an IMU and GNSS. This dataset includes data from two biological dogs across 12 trajectories, sampled at 125 Hz. The second is a larger, publicly available Legged Robot Dataset (the GrandTour dataset), which provides 116 minutes of data from an ANYmal quadruped robot. This robotic dataset includes 49 trajectories featuring high-precision RTK-GNSS ground truth, allowing for a robust comparison between biological biomechanics and engineered robotic locomotion.
</div>
<br>

<div align="center">
  <img width="1335" height="525" alt="dog_setup" src="https://github.com/user-attachments/assets/d82d690b-4f2d-4757-aba4-e9679ebe443d" />
</div>

# Algorithm

* **Model-Based DDR:** Weinberg step-length estimation + Madgwick filter heading.
<div align="center">
  <img width="860" height="195" alt="image" src="https://github.com/user-attachments/assets/a3ac79c5-f3c9-44b1-96f6-b758e3c00e03" />
</div>

* **Deep Learning DDR (DL1):** 1D ResNet architecture for velocity and direction regression.
<div align="center">
  <img width="706" height="163" alt="image" src="https://github.com/user-attachments/assets/5c7229d7-fe5f-49a0-9baf-98e71f53033c" />
</div>

* **Deep Learning DDR (DL2):** Hybrid ResNet/Transformer Encoder for enhanced heading estimation.
<div align="center">
<img width="848" height="217" alt="image" src="https://github.com/user-attachments/assets/4b2dd4cc-32db-4cf2-9cd9-a1ae3bd94474" />
</div>






# Citation 
```bibtex
@article{versano2026dog,
  title={Model-Based and Neural-Aided Approaches for Dog Dead Reckoning},
  author={Versano, Gal and Savin, Itai and Klein, Itzik},
  journal={},
  year={},
}
