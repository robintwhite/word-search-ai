# word-search-ai <br>
Solve Wordsearch puzzle from https://thewordsearch.com/ <br>

![wordsearchai](/InAction.gif) <br>
 <br>
 Create conda environment: 
```
conda env create -f wordsearch-ai-env.yml
```
Run with:
```
python main.py
```
<br>

See Medium article: [Wordsearch AI](https://towardsdatascience.com/ocr-and-the-wordsearch-solver-ai-515aeb816bdf?source=friends_link&sk=bfadea7d44656cf135bc1452217e769d)

You will be prompted to take a screenshot with "S", check console after loading Tensorflow. After, do not move the mouse. <br>
Any errors, re-try by making Wordsearch window full-screen. It may have trouble finding the grid, as this is done automatically. <br>
Must be square. Not tested on other Wordsearch sites. <br>