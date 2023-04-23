### Hi there 👋

<!--
**duuuuu1023/duuuuu1023** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->The system's functions consist of three parts:  
  1.Answer Qusestion:  
  This function corresponds to the multiple choice and single choice questions in machine reading.  
  2.Title generation :  
  Generate relevant titles based on pictures and questions.  
  3. Object detection    
  
  The dataset consists of three parts（coco14）:  
  1.The COCO dataset corresponds to the caption file。  
  2.VQA data based on the COCO dataset 。This dataset contains only questions and answers.   
  3.COCO object detection dataset .  
  
  
  Data processing consists of the following steps:  
  1. Manually annotate 1,000 title and answer corresponding data. The titles and answers correspond as much as possible. If the 5 titles and question answers corresponding to the picture are not appropriate, you can customize the correct title.  processdata.py
  2. Run the bert_train.py  program to train the BERT network model of sentence pairs (title and question correspondence)  
  3. Use the trained weights to predict 10,000 title and question pairs. run bert_predict.py
  4. Among the predicted 10,000 data, selcet the correct data.   Run the bert_train.py and bert_predict.py
  5. Retrain the BERT model and save the optimal results. Finally, generate data in the dataset (130,000 data were predicted in this article).
  6. Perform dataset splitting. 
 
  Run train.py for model training. 


