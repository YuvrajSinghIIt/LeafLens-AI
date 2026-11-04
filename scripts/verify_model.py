from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import WEIGHTS,METRICS,CLASSES
from model_utils import load_model,predict
from PIL import Image
assert WEIGHTS.is_file(),f'Missing weights {WEIGHTS}'
assert METRICS.is_file(),f'Missing metrics {METRICS}'
metrics=json.loads(METRICS.read_text())
assert metrics['classes']==CLASSES
model=load_model()
scores=predict(model,Image.new('RGB',(224,224),(100,160,80)))
assert len(scores)==5 and abs(sum(x[1] for x in scores)-1)<1e-5
print('PASS: model loads, class ordering matches, five predictions sum to 1')
print('Saved held-out test accuracy:',metrics['accuracy'])
