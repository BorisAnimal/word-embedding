# Word2vec embedding implementation

### [Report](https://hackmd.io/W22zZUmyTGWNau0k0P5WiA?both)

Dataset taken from [here](http://pcai056.informatik.uni-leipzig.de/downloads/corpora/eng_wikipedia_2007_1M.tar.gz)

Google analogy test set taken from [here](https://github.com/vecto-ai/word-benchmarks/blob/9b61d0e067e2e68af22160d09450f82d9544cd70/word-analogy/monolingual/en/)

1. Unpack from ```eng_wikipedia_2007_1M.tar.gz``` file ```eng_wikipedia_2007_1M-sentences.txt``` to ```data/``` folder.
2. Run model training:
    ```
    python main.py
    ```
3. Evaluate metrics by 
    * Jupyter notebook (```metrics.ipynb```)   
    * Python script
    ```
   python metrics.py
    ```
