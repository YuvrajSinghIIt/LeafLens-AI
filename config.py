from pathlib import Path
ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/'models'/'leaflens_tomato.pt'
METRICS=ROOT/'models'/'metrics.json'
CLASSES=['Bacterial spot','Early blight','Healthy','Late blight','Leaf mold']
DETAILS={
'Bacterial spot':('🟠','Small dark irregular spots','Inspect adjacent leaves; seek local agricultural guidance.'),
'Early blight':('🍂','Brown lesions, often with concentric rings','Inspect older leaves and consult crop-care advice.'),
'Healthy':('🌿','Relatively even leaf coloration','Continue routine crop monitoring.'),
'Late blight':('⚠️','Dark or water-soaked lesions','Seek prompt expert assessment.'),
'Leaf mold':('🌱','Yellow patches, sometimes mold beneath','Check leaf undersides and ventilation.')}
