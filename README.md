# 🧻 Papers With Topic
_This repository/project was done for XAI501 2021 Spring._

## Intro
More and more AI papers are published every day and we AI developers not only need to read them, but also have to organize all these stuffs to somewhere in order to get grasp of what is going on.
<br>
I started this project to organize papers in some sort of way.

## How to use
### Inference
1. ```pip install -r requirements.txt ``` and probably won't work for sure. Just have below installed correctly. Versions... are not quite dependent so just install whichever version you want it will be fine.
    - `numpy`
    - `sklearn`
    - `pandas`
    - `torch`
    - `transformers` (huggingface 🤗)
    - `easydict`
    - `pyyaml`
    - `gensim` (if you need to use `fasttext` to pretrain word embeddings)
2. ``` python server.py ```
    - Since I'm not using the server, you need to locally host the server.
3. With you internet browser open, go to `localhost:6010` (why 6010? Don't ask)
4. Type in any paper you want, and see how your result goes

Demo video attached below.<br>
https://user-images.githubusercontent.com/62973169/121147618-fca2ea80-c87b-11eb-9e6a-35a76d9ef020.mp4
<br>


### Train
```
import wandb
from paperswithtopic.config import load_config
from paperswithtopic.run import run

cfg = load_config()
run(cfg)
```
Change configurations if needed. Options in taste might be ...
- `model_name`: I have
  - Traditional Sequential models: `rnn`, `lstm`, `gru`
  - Transformer models: `bert`, `bertclassification`, `albert`, `albertclassification`, `electra`, `electraclassification`
    - `xlm`, `xlmclassification` does not work currently
    - `lstmattn` not implemented yet.
    - Difference between naive transformer models and classification (such as `bert` vs. `bertclassification`) is -
      - `bert` feedforwards last hidden state to FC so on so forth, while ...
      - `bertclassification` directly outputs logits
      - 
## Further works   
There are so much more things can be done here such as ...
- If the user not satisfied with their result, they can give a correct selection on their own and retrain the model.
- Show probabilities above 3% (set a threshold)
- 
