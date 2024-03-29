# AutoCoreset: An Automatic Practical Coreset Construction Framework [ICML 2023]


**[1 CSAIL, MIT](https://www.csail.mit.edu/)**  | **[2 DataHeroes](https://dataheroes.ai/)**  | **[3 Rice University](https://www.rice.edu/)**

*[Alaa Maalouf](https://scholar.google.com/citations?user=6r72e-MAAAAJ&hl=en), [Murad Tukan](https://scholar.google.com/citations?user=721xaz0AAAAJ&hl=en), [Vladimir Braverman](https://scholar.google.com/citations?user=DTthB48AAAAJ&hl=en), and [Daniela Rus](https://danielarus.csail.mit.edu/)*


* Paper Link: https://arxiv.org/abs/2305.11980

* Explainer video: https://icml.cc/virtual/2023/poster/24432


![AutoCoreset design](GithubImages/AutoCore_Teaser.png?raw=true)

A coreset is a small weighted subset that approximates the loss function on the whole data, prevalent in machine learning for its advantages. However, current construction methods are problem-dependent and may be challenging for new researchers.

No worries, we got you. We propose *AutoCoreset*: an automatic practical framework for constructing coresets requiring only input data and the cost function (without any other user computation or calculation), making it user-friendly and applicable to various problems. See our open-source code which supports future research and simplifies coreset usage.


# Usage

To use AutoCoreset on your data and desired ML model: 

(1) Modify Line - 134 to be a list containing the path to your Dataset.

(2) Modify Line - 138 to be the name of your ML model (currently we support 'k_means', 'logistic_regression',  'linear_regression', 'svm'). 

(3) You can certainly change the ML model as you wish - as long as you provide the "fit" and "score" functions. 

(4) Run: python main.py

# Citation

If you find this work helpful please cite us:

    @article{maalouf2023autocoreset,

          title={AutoCoreset: An Automatic Practical Coreset Construction Framework},
  
          author={Maalouf, Alaa and Tukan, Murad and Braverman, Vladimir and Rus, Daniela},
  
          journal={arXiv preprint arXiv:2305.11980},
  
          year={2023}

    }
