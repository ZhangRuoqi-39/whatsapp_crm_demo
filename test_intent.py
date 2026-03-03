import sys 
sys.path.insert(0, 'src') 
from intent import IntentClassifier 
c = IntentClassifier() 
r = c.classify('payment failed') 
print(r) 
