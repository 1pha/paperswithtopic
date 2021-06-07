import torch

from flask import Flask, render_template
from flask import request

from paperswithtopic.config import load_config
from paperswithtopic.inference import inference_with_model, revert2class
from paperswithtopic.model import load_model

 
app = Flask(__name__)

# PRELOAD MODEL
cfg = load_config('./asset/config.yml')
model, cfg.device = load_model(cfg)
model_path = './asset/bertclassification_EP12_VALAUC92.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/test', methods=['POST'])
def test():

    paper = request.json # WILL GET string paper name

    preds = inference_with_model(cfg, paper, model)
    fields, probas = revert2class(preds, cfg) # FOR NOW, ONLY SINGLE PAPER
    cvt_pred = {
        f'top{i+1}': (field, int(proba*100)) for i, (field, proba) in enumerate(zip(fields[0], probas[0]))
    }
    print(cvt_pred)
    
    return cvt_pred


@app.route('/get_score', methods=['POST'])
def get_score():

    '''
    1. PREPROCESS
    2. INFERENCE
    3. TO JSON
        - IN THE FORM OF
        - {
            'top1': (field1, proba1),
            'top2': (field2, proba2),
            'top3': (field3, proba3),
        }
    '''

    data = request.json
    print(data)
    user_data = []
    print(data)
    for d in data:
        if 'answer' in d:
            row = [d['assess_id'], d['test_id'],d['tag'], d['answer']]
            user_data.append(row)
     
    print(user_data)
    # score = inference.inference(user_data)
    # score = int(score)
    # # WILL RETURN IN FORM OF
    score = 100
    
    return str(score)

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=6009, debug=True)