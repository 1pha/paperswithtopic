from flask import Flask, render_template
from flask import request

from paperswithtopic.config import load_config
from paperswithtopic.inference import inference, revert2class

 
app = Flask(__name__)

 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/test', methods=['POST'])
def test():
    paper = request.json # WILL GET string paper name

    cfg = load_config()
    cfg.use_saved = False
    cfg.pre_embed = False
    cfg.use_bert_embed = False
    cfg.vocab_size = 30562

    cfg.model_name = 'albertclassification'
    model_path = './models/20210605-2023_albertclassification/albertclassification_EP8_VALAUC89.pth'
    preds = inference(cfg, paper, model_path)
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

    app.run(host="0.0.0.0", port=6008, debug=True)