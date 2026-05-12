# Wise Well: A Healthcare Chatbot

## Description

**Wise Well** is an AI-driven healthcare chatbot developed using the **BioBART v2** NLP framework. Designed for the biomedical domain, the chatbot provides contextually accurate and timely responses to medical queries. It tackles tasks like **Question Answering (QA)**, **Summarization**, **Entity Linking (EL)**, and **Named Entity Recognition (NER)** to enhance accessibility, scalability, and reliability in healthcare.

This project addresses challenges in healthcare accessibility, particularly in underserved regions, and demonstrates how AI can bridge the gap between patients and healthcare providers.
Here is Full Detailed Analysis of my Project [->](https://drive.google.com/file/d/1ucEQ1LRTqdAbWjcJbxGy9SoE2CgrBLEN/view?usp=drive_link)

---

## Lite Version of Wise Well Chatbot

Due to huge resource requirement for this model to be deployed I have made lite version of my project which is deployed here [Wise-Well-Chatbot-Lite](https://46kclo66ry2ptmxgwckb6ben4a0mdfsq.lambda-url.eu-west-2.on.aws/?p=wiswell). I have also made a github repository for this lite version which is the [Wise-Well-Chatbot-Lite-Repo](https://github.com/VatsalSangani/Wise_Well_Chatbot_Lite) for you to understand the logic behind that version.

---

## Model Availability

I have made my model available to Hugging Face Models library which you can access from [Wise-Well-Chatbot-Model](https://huggingface.co/brendvat/BioBARTv2_wisewell_chatbot).

---

## Features

### Core Functionalities
1. **Question Answering (QA)**:
   - Accurate responses to complex medical queries.
   - Fine-tuned on datasets like **BioASQ** and **MedQuAD**.

2. **Summarization**:
   - Condenses extensive medical texts into concise, meaningful summaries.
   - Trained on **iCliniq** and **HealthcareMagic** datasets.

3. **Entity Linking (EL)**:
   - Maps medical entities to standardized vocabularies (e.g., SNOMED-CT).
   - Fine-tuned on **AskAPatient** and **CADEC** datasets.

4. **Named Entity Recognition (NER)**:
   - Identifies and classifies biomedical entities like diseases and medications.
   - Leveraged the **GENIA** dataset for training.

### Key Features
- **Real-Time Interaction**:
  - Asynchronous processing for instant responses.
- **User-Friendly Interface**:
  - Designed with HTML, CSS, and JavaScript for seamless interactions.
- **Scalable Design**:
  - Backend powered by **FastAPI**, ready for cloud deployment.

---

## Technologies Used

### Frameworks and Libraries
- **Python**:
  - NLP: Transformers (Hugging Face), BioBART v2, BioBERT.
  - Backend: FastAPI.
- **Frontend**:
  - HTML, CSS, JavaScript.

### Datasets
- **BioASQ**, **MedQuAD**: For QA tasks.
- **iCliniq**, **HealthcareMagic**: For Summarization.
- **AskAPatient**, **CADEC**: For Entity Linking.
- **GENIA**: For NER.

---

## Results
- **Question Answering (QA)**
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/QA%20Results%20.png)
- **Summarization**
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/Summarization%20Results.png)
- **Entity Linking (EL)**
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/EL%20Results.png)
- **Named Entity Recognition (NER)**
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/NER%20Result.png)
 
---

## User Interface Screenshots
- **Asking Consent**
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/UI%20SS%201.png)
- **Conversation**
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/UI%20SS%207.png)
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/UI%20SS%203.png)
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/UI%20SS%204.png)
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/UI%20SS%205.png)
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/UI%20SS%206.png)
  - ![Alt Text](https://github.com/VatsalSangani/wisewellchatbot/blob/main/UI%20SS%202.png)

